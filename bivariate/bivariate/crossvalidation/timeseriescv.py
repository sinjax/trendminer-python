import inspect
from pylab import *
from IPython import embed

def _slice(parts,dir):
	if dir is "row":
		return (parts,slice(None,None))
	else:
		return (slice(None,None),parts)

class FoldPart(object):
	"""Given a particular array extract various parts according to a fold"""
	def __init__(self, arr, fold, dir="row",valsplit=2./3.,slicefunc=_slice):
		super(FoldPart, self).__init__()
		if arr is None: return
		self.train = arr[slicefunc(fold.train(),dir)];
		self.test = arr[slicefunc(fold.test(),dir)];
		val = fold.val()
		split = int(len(val) * valsplit)
		self.val_param = arr[slicefunc(val[:split],dir)];
		self.val_it = arr[slicefunc(val[split:],dir)];
		self.train_all = arr[slicefunc(fold.train() + val[:split],dir)];
		
	
	def apply(self,fnc,*args,**xargs):
		fp = FoldPart(None,None)
		fp.train = fnc(self.train,*args,**xargs)
		fp.train_all = fnc(self.train_all,*args,**xargs)
		fp.test = fnc(self.test,*args,**xargs)
		fp.val_param = fnc(self.val_param,*args,**xargs)
		fp.val_it = fnc(self.val_it,*args,**xargs)
		return fp
	def apply_inplace(self,fnc,*args,**xargs):
		self.train = fnc(self.train,*args,**xargs)
		self.train_all = fnc(self.train_all,*args,**xargs)
		self.test = fnc(self.test,*args,**xargs)
		self.val_param = fnc(self.val_param,*args,**xargs)
		self.val_it = fnc(self.val_it,*args,**xargs)
		return self
	
	
		

class Fold(object):
	"""A single fold"""
	def __init__(self, train, test, validation):
		super(Fold, self).__init__()
		self._training = train
		self._test = test
		self._validation = validation

	def train(self):
		return self._training

	def train_all(self):
		return self._training + self._validation
	def test(self):
		return self._test
	def val(self):
		return self._validation
	def parts(self,arr,**xargs):
		return FoldPart(arr,self,**xargs)
	def __str__(self):
		return "\n".join([
			"tra: %s"%str(self._training),
			"tes: %s"%str(self._test),
			"val: %s"%str(self._validation)
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
		train = list(set(train) - set(validation))
		self.cstep += 1
		return Fold(train,test,validation)


def tsfi(datalen,**xargs):
	return TimeSeriesFoldIterator(datalen,**xargs);
