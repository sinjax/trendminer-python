from pylab import *
from scipy import sparse as ssp
import spams
from IPython.core.debugger import Tracer
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

# Let's flatten X and make it sparse

# This is X optimised for U optimisation (i.e. W multiplication)
Xu = ssp.csr_matrix(X.reshape([N * U * R, W]))
# Xu now has rows which contain users for each region for each day in that ORDER
# so the rows of Xu are all the users for a region, then all the regions for a day, then all the days

# This is X optimised for W optimisation (i.e. U multiplication)
Xw = X.transpose([1,0,3,2]).reshape([R*N*W,U])
# Xw now has rows which contain words for each day for each region in that ORDER
# so the rows of Xw are all the words for a day, then all the days for a region


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
		tot += max(abs(weights[ind.todense() > 0,:]))
	return tot
# def regulL1L2(weights):
# 	tot = 0
# 	# <demo> --- stop ---
# 	weights.times(weights)

# 	return tot
for epoch in range(10):
	# phase 1: learn u & b given fixed w
	# Here we make V = region stacked (d x u) x t
	NL = R * U
	V = [[None] * N] * R
	for d in range(N):
		for r in range(R):
			rowst = r * U + d * NL
			rowen = (r+1) * U + d * NL
			# V[r][d] = X[rowst:rowen,:].dot(ssp.csc_matrix(w_hat[r,:,:].T)).T
			V[r][d] = ssp.csr_matrix(w_hat[r,:,:]).dot(Xu[rowst:rowen,:].T)
	Vdense = np.diagonal(
		np.tensordot(X,w_hat,axes=([3],[2])),
		axis1=1, axis2=3
	)
	
	spamsParams['loss'] = "square-missing"
	for r in range(R):
		# this is (N x U x T)
		Yr = Y[:,:,r]
		Vrstack = ssp.vstack(V[r],format="csc")
		Yrstack = zeros((T*N,T)) + nan
		for t in range(T):
			Yrstack[t*N:(t+1)*N,t:t+1] = Yr[:,t:t+1]
			
		Yspams = asfortranarray(Yrstack)
		Xspams = Vrstack

		ur0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
		# ur0 = ssp.csr_matrix((Xspams.shape[1],Yspams.shape[1]))
		ur = spams.fistaFlat(
			Yspams,
			Xspams,
			ur0,
			False,**spamsParams
		)

		u_hat[r,:,:] = ur.T

		# Tracer()()

	# phase 2:
	# this is: T x N x W x R
	D = np.diagonal(np.tensordot(u_hat, X, axes=([2],[2])), axis1=3, axis2=0)
	Yest = diagonal(diagonal(w_hat.dot(D),axis1=1,axis2=2),axis1=0,axis2=2)
	flatwhat = array([w_hat.transpose([0,1,2]).flatten()]).T
	errorBeforeWordOpt = norm(Yest - Y)/2 + regulGroups(flatwhat,groups_var) * graphparams['lambda1']
	print "(1) User Opt Error: ",errorBeforeWordOpt

	# Dstack = zeros((N*R*T,W*R*T))
	# THIS IS JUST TO SEE IF IT IS THE SAME!!!!
	Dstack_dense = zeros((N*R*T,W*R*T))
	i = 0;
	for r in range(R):
		for t in range(T):
			Dstack_dense[i*N:(i+1)*N,i*W:(i+1)*W] = D[t,:,:,r]
			i+=1 
	# just to see if it is the same, not to be used in ernest!!!


	Dstack = ssp.lil_matrix((N*R*T,W*R*T))

	i = 0;
	NW = N * W
	for r in range(R):
		# I expect this to be (N * W) * U
		Xwr = Xw[r*NW:(r+1)*NW,:]
		for t in range(T):
			# I expect this to be (N * W) * 1
			Xwrt = Xwr.dot(u_hat[r,t,:])
			for n in range(N):
				DSn = (i * N + n)
				DSw = (i * W )
				Dstack.rows[DSn] = range(DSw,DSw + W)
				Dstack.data[DSn] = Xwrt[n*W:n*W + W]
			i+=1 

	Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
	Yspams = asfortranarray(Yntrflat)
	Xspams = Dstack.tocsc()
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
