class OnlineLearner(object):
	"""
	An online learner can process novel (X,Y) pairs and predict 
	Y based on X.
	
	The process function accepts a list of n "X" instances 
	and a "Y" instance with dimensionality (n x t) where t
	is the number of tasks being predicted.

	The process function can also accept a single "X" instance with
	a "Y" instance with dimensionality (1 x t).


	"""
	def __init__(self):
		super(OnlineLearner, self).__init__()
	def process(self,X,Y): pass
	def predict(self,X): pass
	def _wsparcity(self):
		return self._sparcity(self.w)
	def _usparcity(self):
		return self._sparcity(self.u)

	def _sparcity(self,m,thresh=0):
		return (float((m.toarray()<=thresh).sum())/multiply(*m.shape))