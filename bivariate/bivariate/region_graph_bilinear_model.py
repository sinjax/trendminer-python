from pylab import *
from scipy import sparse as ssp
import spams
from IPython import embed


np.random.seed(1)
# The RGB model, mocked up on synthetic data

R = 3   # regions
T = 5   # tasks aka number of outputs for each region
U = 7   # number of users per region, assumed constant and disjoint
W = 11   # words in vocabulary
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

# add some noise to make things more fun
Y += np.random.random(Y.shape)
# Y has shape N x T x R

# now to learn the fucker

# initialise estimates
u_hat = np.ones_like(u)
w_hat = np.ones_like(w)
b_hat = np.ones_like(b)

print "Data created"

spamsParams = {
	# "intercept": True,
	"loss":"square-missing",
	"compute_gram":False,
	"regul":"l1l2",
	# 'numThreads' : 1,
	'lambda1' : 0.5, 
	# 'it0' : 10, 
	# 'max_it' : 200,
	# 'L0' : 0.1, 
	# 'tol' : 1e-3,
	# 'pos' : False
}
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
	"regul":"graph-ridge",
	'lambda1' : 0.5,
	'verbose' : False
}

def regulGroups(weights,groups):
	tot = 0
	for g in range(groups.shape[1]):
		ind = groups_var[:,g:g+1]
		tot += np.max(np.abs(weights[array(ind.todense())[:,0] > 0,:]))
	return tot
def regulL1L2(weights):
	tot = 0
	# <demo> --- stop ---
	weights.times(weights)

	return tot
for epoch in range(10):
	# phase 1: learn u & b given fixed w
	V = np.diagonal(
		np.tensordot(X,w_hat,axes=([3],[2])),
		axis1=1, axis2=3
	)
	spamsParams['loss'] = "square-missing"
	for r in range(R):
		# this is (N x U x T)
		Vr = V[:,:,:,r]
		Yr = Y[:,:,r]

		Vrstack = vstack([Vr[:,:,t] for t in range(T)])
		Yrstack = zeros((T*N,T)) + nan
		for t in range(T):
			Yrstack[t*N:(t+1)*N,t:t+1] = Yr[:,t:t+1]
			
		Yspams = asfortranarray(Yrstack)
		Xspams = asfortranarray(Vrstack)

		ur0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
		ur = spams.fistaFlat(
			Yspams,
			Xspams,
			ur0,
			False,**spamsParams
		)

		u_hat[r,:,:] = ur.T
	# phase 2:
	# this is: T x N x W x R
	D = np.diagonal(np.tensordot(u_hat, X, axes=([2],[2])), axis1=3, axis2=0)
	Yest = diagonal(diagonal(w_hat.dot(D),axis1=1,axis2=2),axis1=0,axis2=2)
	flatwhat = array([w_hat.transpose([0,1,2]).flatten()]).T
	errorBeforeWordOpt = norm(Yest - Y)/2 + regulGroups(flatwhat,groups_var) * graphparams['lambda1']
	print "(1) User Opt Error: ",errorBeforeWordOpt

	Dstack = zeros((N*R*T,W*R*T))
	i = 0;
	for r in range(R):
		for t in range(T):
			Dstack[i*N:(i+1)*N,i*W:(i+1)*W] = D[t,:,:,r]
			i+=1 
	Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
	Yspams = asfortranarray(Yntrflat)
	Xspams = asfortranarray(Dstack)
	wr0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
	wr = spams.fistaGraph(
		Yspams, Xspams, wr0, graph,False,**graphparams
	)
	w_hat = wr.reshape([R,T,W])
	Yest = diagonal(diagonal(w_hat.dot(D),axis1=1,axis2=2),axis1=0,axis2=2)
	errorAfterWordOpt = norm(Yest - Y)/2 + regulGroups(wr,groups_var) * graphparams['lambda1']
	print "(2) Word Opt Error: ",errorAfterWordOpt
	if errorBeforeWordOpt < errorAfterWordOpt:
		print "THERE WAS A RISE IN ERROR AFTER OPTIMISING WORDS" 
	# Vprime is T x N x W x R
	# now solve for w_hat[:,:,:] and b_hat[:,:]
	# using spams with 
	#   L_21 term over w_hat[ri,:,wi] for all ri, wi	AND
	#   L_21 term over w_hat[:,ti,wi] for all ti, wi 
