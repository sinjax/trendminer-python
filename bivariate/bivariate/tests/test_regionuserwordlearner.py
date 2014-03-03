from pylab import *
from scipy import sparse as ssp

from IPython import embed
import time
from ..learner.batch.regionuserwordlearner import SparseRUWLearner as SRUWLearner
from ..learner.spamsfunc import *
def test_random():
	np.random.seed(1)
	# The RGB model, mocked up on synthetic data

	R = 3   # regions
	T = 5   # tasks aka number of outputs for each region
	U = 103   # number of users per region, assumed constant and disjoint
	W = 201   # words in vocabulary
	N = 510  # training examples for each region & task

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
	# This is X optimised for U optimisation (i.e. W multiplication)
	Xu = ssp.csc_matrix(X.transpose([1,0,2,3]).reshape([R * N * U, W]))
	# Xu now has rows which contain users for each day for each region in that ORDER
	# so the rows of Xu are all the users for a day, then all the days for a region, then all the regions

	# This is X optimised for W optimisation (i.e. U multiplication)
	Xw = X.transpose([1,0,3,2]).reshape([R*N*W,U])
	# Xw now has rows which contain words for each day for each region in that ORDER
	# so the rows of Xw are all the words for a day, then all the days for a region



	learner = SRUWLearner(prep_uspams(), prep_wspams(U,W,N,T,R))
	learner.learn(Xu,Xw,Y)

def prep_uspams():
	spamsParams = {
		"loss":"square-missing",
		"compute_gram":False,
		"regul":"l1l2",
		'lambda1' : 0.5, 
	}
	return FistaFlat(**spamsParams)
def prep_wspams(U,W,N,T,R):
	# set up the group regul
	wrindex = arange(R*T*W).reshape([R,T,W])
	ngroups = W * (T + R)
	eta_g = ones(ngroups,dtype = np.float)
	groups = ssp.csc_matrix(
		np.zeros(
			(ngroups,ngroups),dtype = np.bool
		),dtype = np.bool
	)
	groups_var = zeros([W*R*T,ngroups],dtype=np.bool)
	i = 0
	for word in range(W):
		for r in range(R):
			groups_var[wrindex[r,:,word],i] = 1
			i+=1
		for t in range(T):
			groups_var[wrindex[:,t,word],i] = 1
			i+=1
	groups_var = ssp.csc_matrix(groups_var,dtype=np.bool)
	graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
	graphparams = {
		"loss":"square",
		"regul":"graph",
		'lambda1' : 0.5,
		'verbose' : False
	}

	return FistaGraph(graph,**graphparams)


if __name__ == '__main__':
	test_random()