import spams
from pylab import *
import scipy.sparse as ssp
from IPython import embed

class SpamsFunctions(object):
	"""Setup a spams function and call it"""
	def __init__(self, missing=True,init_strat=None,params=None):
		super(SpamsFunctions, self).__init__()
		self.params = params
		self.init_strat = init_strat
		if missing:
			if self.params["loss"] is "square":
				self.params["loss"] = "square-missing"
		if not self.init_strat:
			self.init_strat = lambda x,y:zeros((x.shape[1],y.shape[1]))

	def call(self,x,y):
		x,y,w0 = self.prepall(x,y)
		w = self._call(x,y,w0)
		b = None
		if self.intercept():
			b = w[-1:,:]
			w = w[:-1,:]
		return w,b
	def _call(self,x,y,w0):
		raise Exception("Not defined")
	def initw(self,x,y):
		w0 = self.init_strat(x,y)
		return np.asfortranarray(w0)
	def intercept(self):
		return "intercept" in self.params and self.params["intercept"]

	def init(self,x,y):
		if self.intercept():
			x = self.hstack(( 
				x, 
				ones( (x.shape[0],1) ) 
			))
		return x,y
	def hstack(self,mats):
		if any([ssp.issparse(x) for x in mats]):
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
		super(FistaFlat, self).__init__(params=xargs)
	def _call(self,x,y,w0):
		w = spams.fistaFlat(y,x,w0,False,**self.params)
		return w
	def __str__(self):
		return "<fistaFlat loss=%s,regul=%s>"%(self.params["loss"],self.params["regul"])

class FistaTree(SpamsFunctions):
	"""docstring for FistaTree"""
	def __init__(self, tree=None,**xargs):
		super(FistaTree, self).__init__(params=xargs)
		self.tree = tree

	def _call(self,x,y,w0):
		w = spams.fistaTree(y,x,w0,self.tree,False,**self.params)
		return w

	def __str__(self):
		return "<fistaTree loss=%s,regul=%s>"%(self.params["loss"],self.params["regul"])