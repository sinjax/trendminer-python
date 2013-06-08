import sys
import spams
from pylab import *
import logging;logger = logging.getLogger("root")
from onlinelearner import OnlineLearner
import scipy.sparse as ssp
import inspect
from bivariate.evaluator.bilineareval import SquareEval as BiSquareEval
from bivariate.evaluator.lineareval import SquareEval
from bivariate.learner.paramlearn import LambdaSearch
from IPython import embed


class BatchBivariateLearner(OnlineLearner):
	"""
	For every X,Y pair, add to an existing set of X,Y and relearn the model 
	from scratch

	All data recieved is collected and recompiled into a new numpy vector
	every time. This gives best conceivable result for a linear system 
	given this optimisation scheme
	"""
	def __init__(self, w_spams_func, u_spams_func, **other_params):
		super(BatchBivariateLearner, self).__init__()
		
		self.allParams = other_params
		self.initDefaults()

		self.elementsSeen = 0
		self.X = None
		self.w = None
		self.u = None
		self.w_bias = None
		self.u_bias = None
		self.bias = None
		self.change_eval = BiSquareEval(self)
		self.part_eval = SquareEval()
		self.w_func = w_spams_func
		self.u_func = u_spams_func
	
	def initDefaults(self):
		self.allParams["bivar_it0"] = self.allParams.get("bivar_it0",3)
		self.allParams["bivar_tol"] = self.allParams.get("bivar_tol",1e-3)
		self.allParams["bivar_max_it"] = self.allParams.get("bivar_max_it",10)

	def predict(self,X):
		pass

	"""
	This function is just a combination of Calling
	setYX,
	then iterating through bivar_max_it iterations of calling
	calculateU and calculateW
	"""
	def process(self,Y,X=None,Xt=None):
		self.setYX(Y,X,Xt)
		bivariter = 0
		sumSSE = 0
		while True:
			logger.debug("Starting iteration: %d"%bivariter)
			bivariter += 1
			U = self.u
			W,w_bias = self.calculateW(U)
			W = ssp.csc_matrix(W)
			U,u_bias = self.calculateU(W)
			self.u = ssp.csc_matrix(U)
			self.w = W
			self.w_bias = w_bias
			self.u_bias = u_bias
			if self.allParams['bivar_max_it'] <= bivariter:
				break
		return sumSSE

	def optimise_lambda(self, fold, lambda_w, lambda_u, Y, X=None, Xt=None):
		yparts = fold.parts(Y).apply(BatchBivariateLearner._expandY)
		
		X,Xt = BatchBivariateLearner._initX(X,Xt)
		ntasks = Y.shape[1]
		ndays = Y.shape[0]
		nusers = X.shape[1]/ndays

		def exp_slice_func(dp,dir):
			parts = []
			for d in dp: 
				dslc = BatchBivariateLearner._rows_for_day(d,ntasks)
				drng = range(dslc.start,dslc.stop)
				parts += [x for x in drng]
			if dir is "row":
				return (parts,slice(None,None))
			else:
				return (slice(None,None),parts)

		u = ssp.csc_matrix(ones((nusers,ntasks)))
		Vprime = BatchBivariateLearner._calculateVprime(X,u,ndays)
		Vprime_parts = fold.parts(Vprime,slicefunc=exp_slice_func)
		

		w_ls = LambdaSearch(self.w_func,self.part_eval)
		w_ls.optimise(lambda_w,Vprime_parts,yparts)
		w,bias = self.w_func.call(Vprime_parts.train_all,yparts.train_all)
		w = ssp.csc_matrix(w)
		Dprime = BatchBivariateLearner._calculateDprime(X,w,ndays)
		Dprime_parts = fold.parts(Dprime,slicefunc=exp_slice_func)

		u_ls = LambdaSearch(self.u_func,self.part_eval)
		u_ls.optimise(lambda_u,Vprime_parts,yparts)
		u = self.u_func.call(Dprime_parts.train_all,yparts.train_all)

		return [(u,self.u_func.params['lambda1']),(w,self.w_func.params['lambda1'])]

	"""
	The number of tasks is the columns of Y
	The number of days is the rows of Y
	The number of users is the columns of X (or the rows of Xt) over the number of days
		Put another way the columns of X contain users batched by days
	The number of words is the rows of X (or the columns of Xt) 
	"""
	def setYX(self,Y,X=None,Xt=None):
		X,Xt = BatchBivariateLearner._initX(X,Xt)
		Y = np.asfortranarray(Y)

		self.X = X
		self.Xt = Xt
		
		self.nusers = X.shape[1]/Y.shape[0]
		self.nwords = X.shape[0]
		self.ntasks = Y.shape[1]
		self.ndays = Y.shape[0]
		self.Yexpanded = self._expandY(Y)

		logger.debug("(ndays=%d,ntasks=%d,nusers=%d,nwords=%d)"%(
			self.ndays,self.ntasks,self.nusers,self.nwords)
		)
		self.u = ssp.csc_matrix(ones((self.nusers,self.ntasks)))
		self.w = ssp.csc_matrix(zeros((self.nwords,self.ntasks)))

	def calculateW(self,U=None):
		if U is None: U = self.u
		Vprime = BatchBivariateLearner._calculateVprime(self.X,U,self.ndays)
		logger.debug("Calling w_func: %s"%self.w_func)
		return self.w_func.call(Vprime,self.Yexpanded)
	
	def calculateU(self,W=None):
		if W is None: W = self.w
		Dprime = BatchBivariateLearner._calculateDprime(self.X,W,self.ndays)
		logger.debug("Calling u_func: %s"%self.u_func)
		return self.u_func.call(Dprime,self.Yexpanded)

	@classmethod
	def _expandY(cls,Y):
		"""
		We expand Y s.t. the values of Y for each task t 
		are held in the diagonals of a t x t matrix whose 
		other values are NaN
		"""
		Yexpanded = ones(
			(
				multiply(*Y.shape),
				Y.shape[1]
			)
		) * nan
		for x in range(Y.shape[1]): 
			ind = x * Y.shape[0]; 
			indnext = (x+1) *Y.shape[0]; 
			Yexpanded[ind:indnext,x] = Y[:,x];
		
		return np.asfortranarray(Yexpanded)
	@classmethod
	def _initX(self,X=None,Xt=None):
		if X is None and Xt is None:
			raise Exception("At least one of X or Xt must be provided")
		if Xt is None:
			Xt = ssp.csc_matrix(X.transpose())
		if X is None:
			X = ssp.csc_matrix(Xt.transpose())
		if not ssp.issparse(X) or not ssp.issparse(Xt):
			raise Exception("X or Xt provided is not sparse, failing")
		return X,Xt


	@classmethod
	def _cols_for_day(cls,d,nusers):
		return slice(d*nusers,(d+1)*nusers)
	@classmethod
	def _rows_for_day(cls,d,ntasks):
		return slice(d*ntasks,(d+1)*ntasks)

	@classmethod
	def _calculateVprime(cls, X, U, ndays):
		logger.debug("Preparing Vprime (X . U)")
		nu = X.shape[1]/ndays
		return ssp.hstack([
			# For every day, extract the day's sub matrix of user/word weights
			# weight each user's words by the user's weight
			# ends up with a (words,days) matrix (csr)
			ssp.hstack([
				X[:,cls._cols_for_day(d,nu)].dot(U[:,t:t+1]) 
				for d in range(ndays)
			],format="csr")
			for t in range(U.shape[1])
		],format="csr").transpose()

	@classmethod
	def _calculateDprime(cls, X, W, ndays):
		logger.debug("Preparing Dprime (X . W)")
		nu = X.shape[1]/ndays
		return ssp.vstack([
			# For every day, extract the day's sub matrix of 
			# user/word weights but now transpose
			# weight each word's users by the word's weight
			# ends up with a (days,user) matrix (csr)
			ssp.hstack([
				X[:,cls._cols_for_day(d,nu)].transpose().dot(W[:,t:t+1])
				for d in range(ndays)
			],format="csc").transpose()
			for t in range(W.shape[1])
		],format="csc")