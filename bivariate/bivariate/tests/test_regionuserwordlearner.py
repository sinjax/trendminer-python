from pylab import *
from scipy import sparse as ssp

from IPython import embed
import time
from ..learner.batch.regionuserwordlearner import SparseRUWLearner as SRUWLearner
from ..learner.batch.regionuserwordlearner import prep_uspams, prep_wspams
# from bivariate import sparse_bias_region_graph_bilinear_model as testcode

def test_random():
	np.random.seed(1)
	# The RGB model, mocked up on synthetic data

	R = 3   # regions
	T = 5   # tasks aka number of outputs for each region
	U = 7   # number of users per region, assumed constant and disjoint
	W = 11   # words in vocabulary
	N = 501  # training examples for each region & task

	# the weights we aim to learn
	u = np.random.random((R, T, U))
	w = np.random.random((R, T, W))
	b = np.random.random((T, R))

	# make them a bit sparse, 
	nri = lambda n,r: array(random(n) * r,dtype=int) # generate some random index
	nw = sum([len(x) for x in w.nonzero()])
	nu = sum([len(x) for x in u.nonzero()])
	# A random set of words to 0 for all regions for all tasks
	w[:,:,nri(W/2,W)] = 0
	u[:,:,nri(U/2,U)] = 0


	# now generate some random training data
	X = np.random.random((N, R, U, W))
	X[np.random.random((N, R, U, W)) < 0.9] = 0

	# construct the response variable y = u X w + b
	Xw = np.diagonal(
		np.tensordot(X,w,axes=([3],[2])),
		axis1=1, axis2=3
	)
	uXw_c = np.tensordot(u,Xw,axes=([2],[1]))
	Y = np.diagonal(
			np.diagonal(
				uXw_c,
				axis1=1,axis2=3)
			,axis1=0,axis2=2) + b
	# add some noise to make things more fun
	Y += np.random.random(Y.shape)
	# This is X optimised for U optimisation (i.e. W multiplication)
	Xu = ssp.csc_matrix(X.transpose([1,0,2,3]).reshape([R * N * U, W]))
	# Xu now has rows which contain users for each day for each region in that ORDER
	# so the rows of Xu are all the users for a day, then all the days for a region, then all the regions

	# This is X optimised for W optimisation (i.e. U multiplication)
	Xw = X.transpose([1,0,3,2]).reshape([R*N*W,U])
	# Xw now has rows which contain words for each day for each region in that ORDER
	# so the rows of Xw are all the words for a day, then all the days for a region

	w_spams = prep_wspams(U,W,T,R,lambda1=0.0005)
	u_spams = prep_uspams(lambda1=0.0005)


	learner = SRUWLearner(u_spams, w_spams, epochs=4,intercept=True, 
		# epoch_callback=check_epoch
	)
	learner.learn(Xu,Xw,Y)

def deepeq(a,b):
	try:
		if type(a) is ndarray:
			ac = np.copy(a)
			bc = np.copy(b)
			nana = isnan(a)
			ac[nana] = 0
			bc[nana] = 0
			return abs(ac - bc).sum() == 0
		elif ssp.issparse(a):
			return abs(a - b).sum() == 0
		elif type(a) is list:
			if not len(a) == len(b): return False
			for x in range(len(a)):
				if not deepeq(a[x],b[x]):
					return False
		elif type(a) is dict:
			if not deepeq(set(a.keys()),set(b.keys())): return False
			for x,v in a.items():
				if not deepeq(v,b[x]):
					return False
		else:
			return a == b
	except Exception, e:
		return False
	return True
	
def check_epoch(epoch):
	test_epoch = testcode.epochout[epoch['epoch']]
	for x in epoch['key_order']:
		if not x in test_epoch: 
			print "%s not found in test_epoch..."%x
			continue
		v = epoch[x]
		vt = test_epoch[x]
		
		if not deepeq(v, vt):
			print "for key: %s"%x
			print "testcode says: %s"%type(vt)
			print "module says: %s"%type(v)
			print "Exiting..."
			embed()
if __name__ == '__main__':
	test_random()