from pylab import *
import spams


# The RGB model, mocked up on synthetic data

R = 6   # regions
T = 3   # tasks aka number of outputs for each region
U = 50   # number of users per region, assumed constant and disjoint
W = 71   # words in vocabulary
N = 100  # training examples for each region & task

# the weights we aim to learn
u = np.random.random((R, T, U))
w = np.random.random((R, T, W))
b = np.random.random((T, R))

# make them a bit sparse
u[np.abs(u) < 0.5] = 0
w[np.abs(w) < 0.5] = 0

# now generate some random training data
X = np.random.random((N, R, U, W))
X[np.abs(X) < 0.5] = 0

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
	'lambda1' : 0.005, 
	# 'it0' : 10, 
	# 'max_it' : 200,
	# 'L0' : 0.1, 
	# 'tol' : 1e-3,
	# 'pos' : False
}

for epoch in range(10):
	# phase 1: learn u & b given fixed w
	V = np.diagonal(
		np.tensordot(X,w_hat,axes=([3],[2])),
		axis1=1, axis2=3
	)
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
	print "(1) Error: ",norm(Yest - Y)

	# this is: (T*R*N)xW
	Dstack = vstack([D[t,:,:,r] for t in range(T) for r in range(R)])

	Ytrnarr = [array([Y[:,t,r]]).T for t in range(T) for r in range(R)]
	Ytrnstack = zeros((T*R*N,T*R)) + nan
	TR = T*R
	for tr in range(TR):
		Ytrnstack[tr*N:(tr+1)*N,tr:tr+1] = Ytrnarr[tr]

	Yspams = asfortranarray(Ytrnstack)
	Xspams = asfortranarray(Dstack)

	wstack0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
	wstack = spams.fistaFlat(
		Yspams,
		Xspams,
		wstack0,
		False,**spamsParams
	)
	w_hat = wstack.reshape((W,T,R)).transpose((2,1,0))

	Yest = diagonal(diagonal(w_hat.dot(D),axis1=1,axis2=2),axis1=0,axis2=2)
	print "(2) Error: ",norm(Yest - Y)
	# Vprime is T x N x W x R
	# now solve for w_hat[:,:,:] and b_hat[:,:]
	# using spams with 
	#   L_21 term over w_hat[ri,:,wi] for all ri, wi	AND
	#   L_21 term over w_hat[:,ti,wi] for all ti, wi 
