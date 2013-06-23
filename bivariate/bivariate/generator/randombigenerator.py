from pylab import *
from IPython import embed
		
class RandomBiGen(object):
	"""docstring for RandomBiGen"""
	def __init__(self, 
		ntasks=3, nusers=5, nwords=10, 
		noise=None, seed=None, wrng=(0.,1.), urng=(0.,1.),
		brng=(0.,0.),wu_sparcity=None
	):
		super(RandomBiGen,self).__init__()
		self.ntasks = ntasks;
		self.nusers = nusers;
		self.nwords = nwords;
		self._noise = noise
		self._w = rand(self.nwords,self.ntasks)
		self._u = rand(self.nusers,self.ntasks)
		self._bias = rand(1,self.ntasks)
		def correct(mat,min,max):
			mat = (mat * (max - min)) + min
			return mat
		self._bias = correct(self._bias,*brng)
		self._w = correct(self._w,*wrng)
		self._u = correct(self._u,*urng)
		if wu_sparcity is not None:
			sparse_words = rand(nwords) < wu_sparcity
			self._w[sparse_words,:] = 0
			sparse_users = rand(nusers) < wu_sparcity
			self._u[sparse_users,:] = 0
		self.nextIndex()

	def nextIndex(self):
		self._x = rand(self.nusers,self.nwords)
		self._y = self._u.T.dot(self._x).dot(self._w) + self._bias
		self._y = diag(self._y).copy()
		if self._noise is not None:
			self._y += self.noise()

	def noise(self):
		return rand(self.ntasks) * self._noise

	def currentWords(self):
		return self._x
	def currentTasks(self):
		return self._y
	
	def generate(self,n=1,include_wu=False,include_bias=False,simple_one=True):
		all_x = []
		all_y = []
		all_w = []
		all_u = []
		all_bias = []
		for x in range(n):
			# ret = [
			# 	self.currentWords(),
			# 	self.currentTasks()
			# ]
			all_x += [self.currentWords()]
			all_y += [self.currentTasks()]
			if include_wu:
				all_w += [self._w]
				all_u += [self._u]
			if include_bias:
				all_bias += [self._bias]
			self.nextIndex()
		if n is 1 and simple_one:
			ret = [all_x[0],all_y[0]]
		else:
			ret = [all_x,all_y]
		if include_wu:
			if n is 1 and simple_one:
				ret += [all_w[0],all_u[0]]
			else:
				ret += [all_w,all_u]
		if include_bias:
			if n is 1 and simple_one:
				ret += [all_bias[0]]
			else:
				ret += [all_bias]
			
		return ret
		