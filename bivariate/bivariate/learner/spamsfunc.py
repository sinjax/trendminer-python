import spams
from pylab import *
import scipy.sparse as ssp
from IPython import embed
import logging;logger = logging.getLogger("root")

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

	def call(self,x,y,w0=None):
		logger.debug("Calling %s"%str(self))
		logger.debug("With x.shape=%s, y.shape=%s"%(str(x.shape),str(y.shape)))
		x,y,w0 = self.prepall(x,y,w0)
		w = self._call(x,y,w0)
		
		b = None
		if self.intercept():
			b = w[-1:,:]
			w = w[:-1,:]
		return w,b
	def _call(self,x,y,w0):
		raise Exception("Not defined")
	def _regul_error(self,w):
		raise Exception("Regul Error is undefined")
	def _error(self,Yest,Y):
		return norm(Yest - Y)/2 
	def error(self,Yest,Y,w):
		return self._error(Yest,Y) + self._regul_error(w)
	def initw(self,x,y,w0):
		if w0 is None:
			w0 = self.init_strat(x,y)
		else:
			if self.intercept():
				w0 = self.vstack((
					w0,
					zeros((1,w0.shape[1]))
				))
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
	def vstack(self,mats):
		if any([ssp.issparse(x) for x in mats]):
			return ssp.vstack(mats,format=mats[0].format)
		else:
			return vstack(mats)
		pass
	def prepall(self,x,y,w0):
		x,y = self.init(x,y)
		w0 = self.initw(x,y,w0)
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

class FistaGraph(SpamsFunctions):
	"""docstring for FistaGraph"""
	def __init__(self,graph,**xargs):
		super(FistaGraph, self).__init__(params=xargs)
		self.graph = graph
	def _call(self,x,y,w0):
		w = spams.fistaGraph(y, x, w0, self.graph,False,**self.params)
		return w
	def __str__(self):
		return "<fistaGraph loss=%s,regul=%s>"%(self.params["loss"],self.params["regul"])
	def _regul_error(self,weights):
		tot = 0
		groups = self.graph["groups_var"]
		for g in range(groups.shape[1]):
			ind = groups[:,g:g+1]
			tot += np.max(np.abs(weights[array(ind.todense())[:,0] > 0,:]))
		return tot * self.params['lambda1']


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