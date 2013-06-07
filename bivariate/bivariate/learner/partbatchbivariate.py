import sys
import spams
from pylab import *
import logging 
from onlinelearner import OnlineLearner
import ipdb;
import scipy.sparse as ssp
import inspect
from bivariate.evaluator.bilineareval import SumSquareEval

# logger = logging.getLogger('bivariate.learners.batchbivariate')
logging.basicConfig(level="DEBUG")
		
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
		self.Q = None
		self.Y = None
		self.w = None
		self.u = None
		self.bias = None
		self.changeEval = SumSquareEval(self)
		self.w_func = w_spams_func
		self.u_func = u_spams_func
	
	def initDefaults(self):
		self.allParams["bivar_it0"] = self.allParams.get("bivar_it0",3)
		self.allParams["bivar_tol"] = self.allParams.get("bivar_tol",1e-3)
		self.allParams["bivar_max_it"] = self.allParams.get("bivar_max_it",10)

	def predict(self,X):
		pass

	"""
	X must be a User*Days X words)
	"""
	def process(self,X,Y):
		

		Q = self.Q
		Y = self.Y

		def initW():
			nwords = self.Q[0].shape[0]
			W = self.initStrat((nwords,Y.shape[1]))
			return ssp.csc_matrix(W)
		def initU():
			nusers = Q[0].shape[1]
			U = self.initStrat((nusers,Y.shape[1]))
			return ssp.csc_matrix(U)
		U = initU()
		W = initW()
		bias = None
		

		
		param = self.spamsDict
		bivariter = 0
		Y = np.asfortranarray(Y)
		Yflat = reshape(self.Y, [multiply(*self.Y.shape),1])
		Yflat = np.asfortranarray(Yflat)
		"""
		We expand Y s.t. the values of Y for each task t 
		are held in the diagonals of a t x t matrix whose 
		other values are NaN
		"""
		Yexpanded = ones(
			(
				multiply(*self.Y.shape),
				self.Y.shape[1]
			)
		) * nan
		for x in range(Y.shape[1]): 
			ind = x * Y.shape[0]; 
			indnext = (x+1) *Y.shape[0]; 
			Yexpanded[ind:indnext,x] = Y[:,x];
		
		Yexpanded = np.asfortranarray(Yexpanded)
		oldSSE = sys.float_info.max
		ntasks = Y.shape[1]
		if self.intercept:
			bias = Y[0:1,:]
		while True:
			bivariter += 1
			# W0 = initW()
			# U0 = initU()
			W0 = np.asfortranarray(W.copy().toarray() )
			U0 = np.asfortranarray(U.copy().toarray() )
			Vprime = ssp.vstack([
				ssp.vstack([
					U[:,x:x+1].T.dot(q.T) for q in Q
				]) 
				for x in range(ntasks)
			])
			# ipdb.set_trace()
			if self.intercept:
				Vprime = ssp.hstack([Vprime,ones((Vprime.shape[0],1))])
				W0 = np.asfortranarray(vstack([W0,bias]))
			# Vprime = np.asfortranarray(Vprime)
			Vprime = ssp.csc_matrix(Vprime)

			(W,optim_info) = spams.fistaFlat(Yexpanded,Vprime,W0,True,**param)
			if self.intercept:
				bias = W[-1:,:]
				logging.debug("W bias: %s"%str(bias))
				W = ssp.csc_matrix(W[:-1,:])
			else:
				W = ssp.csc_matrix(W)
			
			Dprime = ssp.vstack([
				ssp.vstack([
					W[:,x:x+1].T.dot(q) for q in Q
				]) 
				for x in range(ntasks)
			])
			if self.intercept:
				Dprime = ssp.hstack([Dprime,ones((Dprime.shape[0],1))])
				U0 = np.asfortranarray(vstack([U0,bias]))
			Dprime = ssp.csc_matrix(Dprime)
			(U,optim_info) = spams.fistaFlat(Yexpanded,Dprime,U0,True,**param)
			logging.debug("U step optim_info:\n%s"%optim_info)
			if self.intercept:
				bias = U[-1:,:]
				logging.debug("U bias: %s"%str(bias))
				U = ssp.csc_matrix(U[:-1,:])
			else:
				U = ssp.csc_matrix(U)
			
			self.u = U
			self.w = W
			self.bias = bias

			sumSSE = self.changeEval.evaluate(self.Q,self.Y)
			logging.debug("This round's sumSSE: %2.9f"%sumSSE)
			improv = abs(oldSSE - sumSSE)
			oldSSE = sumSSE
			# print "%d,%f"%(bivariter,sumSSE)
			if bivariter > self.allParams['bivar_it0'] and\
				(
					bivariter > self.allParams['bivar_max_it']  or\
					improv < self.allParams['bivar_tol']
				): 
				logging.debug("Iteration: "+str(bivariter))
				logging.debug("Improvment: "+str(improv))
				# ipdb.set_trace()
				# logging.debug("W sparcity: %2.2f"%self._sparcity(W))
				# logging.debug("U sparcity: %2.2f"%self._sparcity(U))
				
				break
		return sumSSE

	
if __name__ == '__main__':
	main()