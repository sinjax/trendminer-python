class BilinearEvaluator(object):
	"""Evaluate a bilinear learner"""
	def __init__(self,learner):
		super(BilinearEvaluator, self).__init__()
		self.learner = learner
	
	def evaluate(self,X,Y):
		pass

class SumSquareEval(BilinearEvaluator):
	"""Evaluate using the sum square error"""
	def __init__(self, learner):
		super(SumSquareEval, self).__init__(learner)

	def evaluate(self,X,Y):
		if type(X) is not list:
			X = [X]
		total = 0
		for i in range(len(X)):
			y = Y[i:i+1,:]
			x = X[i].T
			total+=self._sumSquaredError(
				x,y,
				self.learner.u,
				self.learner.w,
				self.learner.bias
			)
		return total
	def _sumSquaredError(self,x,y,u,w,bias=None):
		total = 0
		for t in range(u.shape[1]):
			# ipdb.set_trace()
			dotproduct = u[:,t:t+1].T.dot(x).dot(w[:,t:t+1])[0,0]
			withoutbias = dotproduct
			if bias is not None: dotproduct += bias[0,t]
			# logging.info("task=%d,y=%2.5f,v=%2.5f,vb=%2.5f,delta=%3.9f"%(
			# 	t,y[0,t],dotproduct,withoutbias,
			# 	pow(y[0,t]-dotproduct,2)
			# ))
			total+= pow(y[0,t] - dotproduct,2)
		return total

class MeanSumSquareEval(SumSquareEval):
	"""Evaluate using the sum square error"""
	def __init__(self, learner):
		super(MeanSumSquareEval, self).__init__(learner)

	def evaluate(self,X,Y):
		if type(X) is not list:
			X = [X] 
		return super(MeanSumSquareEval, self).evaluate(X,Y)/(Y.shape[0] * Y.shape[1])
		
class RootMeanSumSquareEval(MeanSumSquareEval):
	"""Evaluate using the sum square error"""
	def __init__(self, learner):
		super(RootMeanSumSquareEval, self).__init__(learner)

	def evaluate(self,X,Y):
		return sqrt(super(RootMeanSumSquareEval, self).evaluate(X,Y))