import spams
from pylab import *
import scipy.sparse as ssp

class SpamsFunctions(object):
	"""Setup a spams function and call it"""
	def __init__(self, init_strat=None,params=None):
		super(SpamsFunctions, self).__init__()
		self.params = params
		self.init_strat = init_strat
		if not self.init_strat:
			self.init_strat = lambda x,y:zeros((x.shape[1],y.shape[1]))

	def call(self,x,y):
		raise Exception("undefined")
	def initw(self,x,y):
		w0 = self.init_strat(x,y)
		return np.asfortranarray(w0)
	def init(self,x,y):
		if "intercept" in self.params and self.params["intercept"]:
			x = self.hstack(( 
				x, 
				ones( (x.shape[0],1) ) 
			))
		return x,y
	def hstack(self,mats):
		if "sparse" in str(type(mats[0])):
			return ssp.hstack(mats,format=mats[0].format)
		else:
			return hstack(mats)
		pass
	def prepall(self,x,y):
		x,y = self.init(x,y)
		w0 = self.initw(x,y)
		return x,y,w0


class FistaFlat(SpamsFunctions):
	"""docstring for FistaFlat"""
	def __init__(self,**xargs):
		super(FistaFlat, self).__init__(**xargs)
	def call(self,x,y):
		x,y,w0 = self.prepall(x,y)
		w = spams.fistaFlat(y,x,w0,False,**self.params)
		return w

class FistaTree(SpamsFunctions):
	"""docstring for FistaTree"""
	def __init__(self, tree=None,**xargs):
		super(FistaTree, self).__init__(**xargs)
		self.tree = tree

	def call(self,x,y):
		x,y,w0 = self.prepall(x,y)
		w = spams.fistaTree(y,x,w0,self.tree,False,**self.params)
		return w
		
		