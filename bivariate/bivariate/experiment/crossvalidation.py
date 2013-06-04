class TimeSeriesFoldIterator(object):
	"""
	A time series fold generator creates Fold instances 
	(i.e. training, test and validation) which involve
	some number of training examples predicting some
	number of test examples which are data values with
	higher indecies than the training examples. In this
	way these folds can simulate data arriving in some fixed 
	chronological order.

	The TimeSeriesFoldIterator starts with some initial number of
	training examples. The test set is then some number of examples
	after this 

	"""
	def __init__(self, 
		datasize, 
		start=None, ntrain=3,nval=1,ntest=1,
		
	):
		super(TimeSeriesFoldIterator, self).__init__()
		self.arg = arg

def tsfi(*args,**xargs):
	return TimeSeriesFoldIterator(*args,**xargs);