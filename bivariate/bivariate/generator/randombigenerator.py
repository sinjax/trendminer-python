from pylab import *
		
class RandomBiGen(object):
	"""docstring for RandomBiGen"""
	def __init__(self, 
		ntasks=3, nusers=5, nwords=10, 
		noise=None, seed=None
	):
		super(RandomBiGen,self).__init__()
		self.ntasks = ntasks;
		self.nusers = nusers;
		self.nwords = nwords;
		self._noise = noise
		self.nextIndex()

	def nextIndex(self):
		self._w = rand(self.nwords,self.ntasks)
		self._u = rand(self.nusers,self.ntasks)
		self._x = rand(self.nusers,self.nwords)
		self._y = self._u.T.dot(self._x).dot(self._w)
		self._y = diag(self._y)
		if self._noise is not None:
			self._y += self.noise()

	def noise(self):
		return rand(self.ntasks) * self._noise

	def currentWords(self):
		return self._x
	def currentTasks(self):
		return self._y
	
	def generate(self,n=1,include_wu=False):
		allret = []
		for x in range(n):
			ret = [
				self.currentWords(),
				self.currentTasks()
			]
			if include_wu:
				ret += [self._w,self._u]
			allret += [ret]
			self.nextIndex()
		return allret
		