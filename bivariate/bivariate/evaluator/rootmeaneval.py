from meaneval import MeanSumSquareEval
from pylab import *
class RootMeanSumSquareEval(MeanSumSquareEval):
	"""Evaluate using the sum square error"""
	def __init__(self, learner):
		super(RootMeanSumSquareEval, self).__init__(learner)

	def evaluate(self,X,Y):
		return sqrt(super(RootMeanSumSquareEval, self).evaluate(X,Y))
		