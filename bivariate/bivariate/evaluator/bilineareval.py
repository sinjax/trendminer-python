class BilinearEvaluator(object):
	"""Evaluate a bilinear learner"""
	def __init__(self,learner):
		super(BilinearEvaluator, self).__init__()
		self.learner = learner
	
	def evaluate(self,X,Y):
		pass