import inspect
from pylab import *

class Fold(object):
	"""A single fold"""
	def __init__(self, train, test, validation):
		super(Fold, self).__init__()
		self.training = train
		self.test = test
		self.validation = validation

	def __str__(self):
		return "\n".join([
			"tra: %s"%str(self.training),
			"tes: %s"%str(self.test),
			"val: %s"%str(self.validation)
		])
	def __indices__(self,len):
		print "bees"


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
	def __init__(self, datalen, start=0, ntraining=7,nvalidation=3,ntest=1,**xargs):
		super(TimeSeriesFoldIterator, self).__init__()
		argspec = inspect.getargspec(self.__init__)
		inargs = list( set(argspec.args) - set(["self"]) )
		for x in inargs: self.__dict__[x] = locals()[x]
		self.__dict__.update(xargs)

		self.cstep = 0;
		self.maxstep = ((datalen - start) - ntraining) / ntest
	def __iter__(self):
		return self

	def next(self):
		if self.cstep >= self.maxstep: raise StopIteration()
		finalTrain = self.start + self.cstep * self.ntest + self.ntraining
		
		train = [ x 
			for x in range(self.start, finalTrain)
		]
		test = [ x 
			for x in range(finalTrain, finalTrain + self.ntest)
		]

		trainMid = len(train)/2
		halfnval = self.nvalidation/2
		validation = train[trainMid-halfnval:(trainMid-halfnval+self.nvalidation)]
		self.cstep += 1
		return Fold(train,test,validation)


def tsfi(datalen,**xargs):
	return TimeSeriesFoldIterator(datalen,**xargs);
