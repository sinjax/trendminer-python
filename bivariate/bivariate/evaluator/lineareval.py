class LinearEvaluator(object):
	"""Evaluate a bilinear learner"""
	def __init__(self):
		super(LinearEvaluator, self).__init__()
	
	def evaluate(self,X,Y,theta,bias=None):
		pass

class SquareEval(LinearEvaluator):
	"""sum of the square difference between y and x . theta + bias"""
	def __init__(self):
		super(SquareEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		for t in range(theta.shape[1]):
			dotproduct = X.dot(theta[:,t:t+1])[0,0]
			if bias is not None: dotproduct += bias[0,t]
			total+= pow(Y[0,t] - dotproduct,2)
		return total

class MeanEval(SquareEval):
	"""The SquareEval divided by the total number of Ys (i.e. tasks and days)
	being estimated"""
	def __init__(self, arg):
		super(MeanEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		err = super(MeanEval,self).evaluate(X,Y,theta,bias)
		return err / (Y.shape[1] * Y.shape[0])

class RootMeanEval(MeanEval):
	"""The square root of the MeanEval"""
	def __init__(self):
		super(RootMeanEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		err = super(RootMeanEval,self).evaluate(X,Y,theta,bias)
		return sqrt(err) 
		
		
		