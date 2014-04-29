from scipy import sparse as ssp
from pylab import *
from IPython import embed
def knnmatrix(W,k):
	n=W.shape[0];
	indi=zeros(k*n,dtype=int);
	indj=zeros(k*n,dtype=int);
	inds=zeros(k*n);
	for ii in range(n):
		o = array(
			argsort(
				-W[ii,:].todense()
				,kind='mergesort'
			)[:,:k]
		)[0,:]
		indi[ii*k:(ii+1)*k]=ii;
		indj[ii*k:(ii+1)*k]=o;
		inds[ii*k:(ii+1)*k]=W[ii,o].todense();
	
	WN=ssp.coo_matrix(
		(inds,(indi,indj)),
		(n,n)
	)
	return WN
