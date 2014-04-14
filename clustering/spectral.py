from util import textscan
from scipy import sparse as ssp
from knnmatrix import knnmatrix
from IPython import embed
from pylab import *
from scipy.stats import mstats as mstats
from sklearn.cluster import spectral_clustering
from centralityn import centralityn

def spectral(tweetfile,npmifile,dictfile,k,noc):
	Ptmp=textscan(npmifile,'([^ ]*) ([^ ]*) ([^ ]*)');
	PP=textscan(dictfile,'(.*) (.*)',(int,str));
	PP[0] -= 1
	PMI=ssp.coo_matrix(
		(Ptmp[2],(Ptmp[0]-1,Ptmp[1]-1)),
		(PP[0].shape[0],PP[0].shape[0])
	).tocsr();

	W=knnmatrix(PMI,k);
	# This is hidious and wrong and it must be fixed
	W=ssp.csr_matrix(minimum(W.todense(),W.T.todense()))
	
	s,comp = ssp.csgraph.connected_components(W,directed=False)
	comp_mode = mstats.mode(comp)[0]
	inds = comp==comp_mode
	inds = [x for x in range(W.shape[0]) if inds[x]]
	WW = W[inds,:][:,inds]
	P=PP[1][inds];

	ids = P;
	X = WW;

	c = spectral_clustering(X,n_clusters=noc, eigen_solver='arpack')
	fid=file("".join(['cl.',tweetfile,'-',str(noc)]),'w');
	for i in range(max(c)+1):
		cl=[x for x in range(len(c)) if c[x] == i]
		b,wordsix = centralityn(cl,X,ids);
		for j in range(len(b)):
			word=wordsix[j];
			fid.write('%s %d %.5f\n'%(word,i,b[j]));
	
