import sys
import spams
from pylab import *
import logging;logger = logging.getLogger("root")
from onlinelearner import OnlineLearner
import scipy.sparse as ssp
import inspect
from bivariate.evaluator.bilineareval import RootMeanSquareEval as BiMeanSquareEval
from bivariate.evaluator.lineareval import RootMeanEval
from bivariate.learner.paramlearn import LambdaSearch
from IPython import embed
import bivariate.experiment.expstate as es


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
		self.change_eval = BiMeanSquareEval(self)
		self.part_eval = RootMeanEval()
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
	def process(self,Y,X=None,Xt=None,tests=None):
		self.setYX(Y,X,Xt)
		bivariter = 0
		sumSSE = 0
		esiter = list()
		es.state()["iterations"] = esiter
		# in the first iteration we calculate W by using ones on U
		U = ssp.csc_matrix(ones(self.u.shape))
		while True:
			esiterdict = dict()
			esiterdict["i"] = bivariter
			logger.debug("Starting iteration: %d"%bivariter)
			bivariter += 1
			W,w_bias,err = self.calculateW(U,tests=tests)
			esiterdict["w"] = W
			esiterdict["w_sparcity"] = (abs(W) > 0).sum()
			esiterdict["w_bias"] = w_bias
			esiterdict["w_test_err"] = err
			if "test" in err: logger.debug("W sparcity=%d,test_total_err=%2.2f,test_err=%s"%(esiterdict["w_sparcity"],err['test']["totalsse"],str(err['test']["diffsse"])))
			W = ssp.csc_matrix(W)
			U,u_bias,err = self.calculateU(W,tests=tests)
			esiterdict["u"] = U
			esiterdict["u_sparcity"] = (abs(U) > 0).sum()
			esiterdict["u_bias"] = u_bias
			esiterdict["u_test_err"] = err
			if "test" in err: logger.debug("U sparcity=%d,test_total_err=%2.2f,test_err=%s"%(esiterdict["u_sparcity"],err['test']["totalsse"],str(err['test']["diffsse"])))
			U = ssp.csc_matrix(U)
			self.u = U
			self.w = W
			self.w_bias = w_bias
			self.u_bias = u_bias
			esiter += [esiterdict]
			if self.allParams['bivar_max_it'] <= bivariter:
				break
		return sumSSE

	def optimise_lambda(self, lambda_w, lambda_u, Yparts, Xparts):
		logger.debug("... expanding Yparts")
		Yparts = Yparts.apply(BatchBivariateLearner._expandY)

		ls = LambdaSearch(self.part_eval)
		ntasks = Yparts.train_all.shape[1]
		ndays = Yparts.train_all.shape[0]/ntasks
		nusers = Xparts.train_all.shape[1]/ndays

		u = ssp.csc_matrix(ones((nusers,ntasks)))
		logger.debug("... Preparing VPrime")
		Vprime_parts = Xparts.apply(
			BatchBivariateLearner._calculateVprime,u
		)
		logger.debug("... Optimising lambda for w")
		ls.optimise(self.w_func,lambda_w,Vprime_parts,Yparts,name="w")
		logger.debug("... Calculating w with optimal lambda")
		w,bias = self.w_func.call(Vprime_parts.train_all,Yparts.train_all)
		w = ssp.csc_matrix(w)
		logger.debug("... Preparing Dprime")
		Dprime_parts = Xparts.apply(
			BatchBivariateLearner._calculateDprime,w,u.shape
		)
		logger.debug("... Optimising lambda for u")
		ls.optimise(self.u_func, lambda_u, Dprime_parts, Yparts,name="u")
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
		self.u = ssp.csc_matrix(zeros((self.nusers,self.ntasks)))
		self.w = ssp.csc_matrix(zeros((self.nwords,self.ntasks)))

	def calculateW(self,U=None,tests=None):
		if U is None: U = self.u
		Vprime = BatchBivariateLearner._calculateVprime(self.X,U)
		logger.debug("Calling w_func: %s"%self.w_func)
		W,w_bias = self.w_func.call(Vprime,self.Yexpanded)
		err = self.part_eval.evaluate(Vprime,self.Yexpanded,W,w_bias)
		testerr = {"train_all":err}
		if tests is not None:
			for testName,(testX,testY) in tests.items():
				testerr[testName] = self.part_eval.evaluate(
					self._calculateVprime(testX,U),
					self._expandY(testY),
					W,w_bias
				)
		return W,w_bias,testerr
	
	def calculateU(self,W=None,tests=None):
		if W is None: W = self.w
		Dprime = BatchBivariateLearner._calculateDprime(self.X,W,self.u.shape)
		logger.debug("Calling u_func: %s"%self.u_func)
		U,u_bias = self.u_func.call(Dprime,self.Yexpanded)
		err = self.part_eval.evaluate(Dprime,self.Yexpanded,U,u_bias)
		testerr = {"train_all":err}
		if tests is not None:
			for testName,(testX,testY) in tests.items():
				testerr[testName] = self.part_eval.evaluate(
					self._calculateDprime(testX,W,self.u.shape),
					self._expandY(testY),
					U,u_bias
				)
		return U,u_bias,testerr

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
	def _user_day_slice(cls,nusers):
		def exp_slice_func(dp,dir):
			parts = []
			for d in dp: 
				dslc = BatchBivariateLearner._cols_for_day(d,nusers)
				drng = range(dslc.start,dslc.stop)
				parts += [x for x in drng]
			if dir is "row":
				return (parts,slice(None,None))
			else:
				return (slice(None,None),parts)
		return exp_slice_func

	"""
	Expects an X such that users are held in the columns
	and a U which weights each user for each task
	"""
	@classmethod
	def _calculateVprime(cls, X, U):
		# logger.debug("Preparing Vprime (X . U)")
		nu = U.shape[0]
		ndays = X.shape[1]/nu
		# stack in the columns the (word,days) matricies for each task
		# so the dimensions are (word,days*tasks).
		# we then transpose such that the days*tasks are in the columns
		# and the words in the rows resulting in (days*tasks,word)
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

	"""
	Expects an X such that users are held in the columns
	and a W which weights each word for each task
	"""
	@classmethod
	def _calculateDprime(cls, X, W, Ushape):
		# logger.debug("Preparing Dprime (X . W)")
		nu = Ushape[0]
		ndays = X.shape[1]/nu
		# stack in the columns the (days,user) matricies for each task
		# so the dimensions are (days*tasks,user).
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

	@classmethod
	def XYparts(self,fold,X,Y):
		Yparts = fold.parts(Y)
		Xparts = fold.parts(
			X,dir="col",
			slicefunc=BatchBivariateLearner._user_day_slice(X.shape[1]/Y.shape[0])
		)
		return Xparts,Yparts