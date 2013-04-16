from sumsquare import SumSquareEval

class MeanSumSquareEval(SumSquareEval):
	"""Evaluate using the sum square error"""
	def __init__(self, learner):
		super(MeanSumSquareEval, self).__init__(learner)

	def evaluate(self,X,Y):
		if type(X) is not list:
			X = [X] 
		return super(MeanSumSquareEval, self).evaluate(X,Y)/(Y.shape[0] * Y.shape[1])
		