import numpy as np

# The RGB model, mocked up on synthetic data

R = 2   # regions
T = 3   # tasks aka number of outputs for each region
U = 5   # number of users per region, assumed constant and disjoint
W = 7   # words in vocabulary
N = 11  # training examples for each region & task

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
        axis1=0,axis2=4
        ),
    axis1=0,axis2=3
) + b

# add some noise to make things more fun
Y += np.random.random(Y.shape)

# Y has shape N x T x R
print Y.shape

# now to learn the fucker

# initialise estimates
u_hat = np.ones_like(u)
w_hat = np.ones_like(w)
b_hat = np.ones_like(b)

for epoch in range(10):
    # phase 1: learn u & b given fixed w
    V = np.diagonal(
        np.tensordot(X,w_hat,axes=([3],[2])),
        axis1=1, axis2=3
    )

    # Vstack ends up with a ((T * N) x (U * R)) matrix
    Vstack = np.vstack([
        np.hstack([
            V[:,:,:,x] for x in range(R)])[:,:,y] 
        for y in range(T)
    ])

    # Y ends up with ((T*R*N) x (T*R))

    # User weighting looks like (U x (T*R))


    # phase 2:
    Vprime = np.diagonal(np.tensordot(u, X, axes=([2],[2])), axis1=3, axis2=0)
    # Vprime is T x N x W x R
    # now solve for w_hat[:,:,:] and b_hat[:,:]
    # using spams with 
    #   L_21 term over w_hat[ri,:,wi] for all ri, wi    AND
    #   L_21 term over w_hat[:,ti,wi] for all ti, wi 
