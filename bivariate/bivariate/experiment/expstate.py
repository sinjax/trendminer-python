import os
from pylab import *
import shutil

_singleton = None

class FakeDict(dict):
	"""docstring for FakeDict"""
	def __init__(self):
		super(FakeDict, self).__init__()
	def __setitem__(self,key,value):
		# do nothing!
		pass
		
class ExperimentState(object):
	"""
	An experiment state can hold variables in a state, flush
	the state to a file, create a new state and so on.

	States are created by giving a name
	the name of the state is also the file output

	The root of all the states must be given wherein
	a .state directory is created and filed
	"""
	def __init__(self, root,fake=False):
		super(ExperimentState, self).__init__()
		self.state_root = os.sep.join([root,".state"])
		self._currentState = None
		self._currentStateName = None
		self._fake = fake
		if not os.path.exists(self.state_root): 
			os.makedirs(self.state_root)


	def nextState(self,name):
		if not self._fake:
			self._currentState = dict()
		else:
			self._currentState = FakeDict()
		self._currentStateName = name

	def function(self):
		pass

	def flush(self):
		if self._currentState is None: return
		outname = os.sep.join([self.state_root,"%s.npy"%self._currentStateName])
		np.save(outname,self._currentState)

	def load_states(self):
		stateFs = os.listdir(self.state_root)
		ret = dict()
		for stateF in stateFs:
			sname,sdict = self.load_state(stateF)
			ret[sname] = sdict
		return ret
	def load_state(self,stateF):
		full = os.sep.join([self.state_root,stateF])
		name = stateF[:-4]
		return name,np.load(full).tolist()

def exp(root="./",fake=False):
	global _singleton
	if _singleton is None: 
		_singleton = ExperimentState(root,fake=fake)
	return _singleton

def state(state=None):
	if state is None and exp()._currentStateName is None:
		raise Exception("No state has been specified")
	if state is not None and exp()._currentStateName is not state:
		exp().nextState(state)
	return exp()._currentState

def flush():
	exp().flush()
def load_states(name=None):
	return exp(name).load_states();
def add(d,*args):
	for x in args:
		if x in d:
			state()[x] = d[x]
