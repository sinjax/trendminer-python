import sys
import spams
from pylab import *
import logging
import util
from onlinelearner import OnlineLearner

logger = logging.getLogger('bivariate.learners.batchbivariate')
		

class BatchBivariateLearner(OnlineLearner):
	"""
	For every X,Y pair, add to an existing set of X,Y and relearn the model from
	scratch

	All data recieved is collected and recompiled into a new numpy vector
	every time. This gives best conceivable result for a linear system 
	given this optimisation scheme
	"""
	def __init__(self, script=None,independantw=True,independantu=True, *spamsArgs, **spamsDict):
		super(BatchBivariateLearner, self).__init__(script=script)
		self.spamsArgs = spamsArgs;
		self.spamsDict = spamsDict;
		self.elementsSeen = 0
		self.Q = None
		self.Y = None
		self.indw = independantw
		self.indu = independantu
		self.initStrat = lambda shape: rand(*shape)
		# self.initStrat = lambda shape: zeros(shape)

	@classmethod
	def declareOutputFields(cls):
		return ["weights","elementsSeen"]

	def calculateSumSSE(self,u,w):
		"""
			Calculates the sum squared error of all seen
			X values for all Ys.

			This calculation is different depending on 
			whether independantu and independantw are set.

			Specifically, if both are true then U' . X_n . W results
			in a (t x t) matrix and so each task must be calculated
			and checked seperately. i.e. for all t: U'_t x X_n . W_t



			This is also the case when one or the other is true and is not the case 
			if both are false in which case u and w are p x 1 and  and so U' . X_n . W
			is also 1xt
		"""
		total = 0
		for i in range(self.Q.shape[0]):
			y = self.Y[i:i+1,:]
			X = self.Q[i,:,:].T
			if self.indw and self.indu:
				for t in range(u.shape[1]):
					dotproduct = u[:,t:t+1].T.dot(X).dot(w[:,t:t+1])
					total += pow(y[:,t:t+1] - dotproduct,2).sum()
			elif self.indw and not self.indu:
				for t in range(u.shape[1]):
					total += pow(y[:,t:t+1] - u.T.dot(X).dot(w[:,t:t+1]),2).sum()
			elif not self.indw and self.indu:
				for t in range(u.shape[1]):
					total += pow(y[:,t:t+1] - u[:,t:t+1].T.dot(X).dot(w),2).sum()
			else:
				total += pow(y - u.T.dot(X).dot(w),2).sum()
		return total
	def process(self,X,Y):
		if len(X.shape) is 3:
			if self.elementsSeen is 0:
				self.Q = X
				self.Y = Y
				elementsSeen = X.shape[0] - 1 # 1 gets added later!
			else:
				self.Q = vstack((self.Q,X))
				self.Y = vstack((self.Y,Y))
				elementsSeen += X.shape[0] - 1 # 1 gets added later!
		else:
			Xn = array([X])
			Yn = array(Y)
			if self.elementsSeen is 0:
				self.Q = Xn
				self.Y = Yn
			else:
				self.Q = vstack((self.Q,Xn))
				self.Y = vstack((self.Y,Yn))

		Q = self.Q
		
		Y = self.Y
		def initW():
			if self.indw:
				W = self.initStrat((self.Q.shape[1],Y.shape[1]))
			else:
				W = self.initStrat((self.Q.shape[1],1))
			logging.debug("Batch init W: \n%s"%str(W))
			return np.asfortranarray(W)
		def initU():
			if self.indu:
				U = self.initStrat((Q.shape[2],Y.shape[1]))
			else:
				U = self.initStrat((Q.shape[2],1))
			logging.debug("Batch init U: \n%s"%str(U))
			return np.asfortranarray(U)
		U = initU()
		W = initW()

		self.elementsSeen+=1
		# (W, optim_info) = spams.fistaFlat(
		# 	self.Y,self.X,W0,True,**self.spamsDict)
		param = self.spamsDict
		bivariter = 0
		Y = np.asfortranarray(Y)
		Yflat = reshape(self.Y, [multiply(*self.Y.shape),1])
		Yflat = np.asfortranarray(Yflat)
		Yexpanded = ones((multiply(*self.Y.shape),self.Y.shape[1])) * nan
		for x in range(Y.shape[1]): 
			ind = x * Y.shape[0]; 
			indnext = (x+1) *Y.shape[0]; 
			Yexpanded[ind:indnext,x] = Y[:,x];
		Yexpanded = np.asfortranarray(Yexpanded)
		oldSSE = sys.float_info.max
		while True:
			bivariter += 1
			# W0 = initW()
			# U0 = initU()
			W0 = np.asfortranarray(W.copy())
			U0 = np.asfortranarray(U.copy())
			if self.indw and self.indu:
				# Section 3.3 of bill's paper
				# V is calculated across tasks + users
				# D is calculated across tasks + words
				if param['loss'] is "square": param['loss'] = "square-missing"
				Vprime = vstack(
					[
						U[:,x:x+1].T.dot(Q.transpose([0,2,1]))[0,:,:] 
						for x in range(Y.shape[1])
					]
				)
				Vprime = np.asfortranarray(Vprime)
				(W,optim_info) = spams.fistaFlat(Yexpanded,Vprime,W0,True,**param)
				Dprime = vstack(
					[
						W[:,x:x+1].T.dot(Q)[0,:,:] 
						for x in range(Y.shape[1])
					]
				)
				Dprime = np.asfortranarray(Dprime)
				(U,optim_info) = spams.fistaFlat(Yexpanded,Dprime,U0,True,**param)

			elif self.indw and not self.indu:
				V = U.T.dot(Q.transpose([0,2,1]))[0,:,:]
				V = np.asfortranarray(V)
				(W,optim_info) = spams.fistaFlat(Y,V,W0,True,**param)
				Dprime = vstack([W[:,x:x+1].T.dot(Q)[0,:,:] for x in range(Y.shape[1])])
				Dprime = np.asfortranarray(Dprime)
				(U,optim_info) = spams.fistaFlat(Yflat,Dprime,U0,True,**param)
			elif self.indu and not self.indw:
				D = W.T.dot(Q)[0,:,:]
				D = np.asfortranarray(D)
				(U,optim_info) = spams.fistaFlat(Y,D,U0,True,**param)
				Vprime = vstack(
					[
						U[:,x:x+1].T.dot(Q.transpose([0,2,1]))[0,:,:] 
						for x in range(Y.shape[1])
					]
				)
				Vprime = np.asfortranarray(Vprime)
				(W,optim_info) = spams.fistaFlat(Yflat,Vprime,W0,True,**param)
			else:
				V = U.T.dot(Q.transpose([0,2,1]))[0,:,:]
				V = np.asfortranarray(V)
				(W,optim_info) = spams.fistaFlat(Y,V,W0,True,**param)
				D = W.T.dot(Q)[0,:,:]
				D = np.asfortranarray(D)
				(U,optim_info) = spams.fistaFlat(Y,D,U0,True,**param)
			
			sumSSE = self.calculateSumSSE(U,W)
			improv = abs(oldSSE - sumSSE)
			oldSSE = sumSSE
			# print "%d,%f"%(bivariter,sumSSE)
			if bivariter > param['it0'] and (bivariter > param['max_it']  or improv < param['tol']): 
				logger.debug("Iteration: "+str(bivariter))
				logger.debug("Improvment: "+str(improv))
				break
		return [[U,W],sumSSE]
		


def main():
	from nose.tools import assert_equal
	spamsDict = {
		"numThreads": 1, "verbose": False,
		"lambda1": 0.01,"lambda2":0.01, "it0": 10, "max_it": 1000,
		"L0":0.1,"tol":1e-3,"intercept" : False, "pos": False
	}
	spamsDict['compute_gram'] = True
	spamsDict["loss"] = "square"
	spamsDict["regul"] = "l1"
	indw = False
	indu = False
	bolt = BatchBivariateLearner(
		script=__file__,
		independantw=indw,
		independantu=indu,**spamsDict)
	

if __name__ == '__main__':
	main()