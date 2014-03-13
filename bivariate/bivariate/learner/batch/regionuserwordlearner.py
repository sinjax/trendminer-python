from copy import deepcopy as dc
import spams
import sys
import logging;logger = logging.getLogger("root")
from pylab import *
import scipy.sparse as ssp
import time
from IPython import embed
from ...tools.utils import reshapeflat, reshapecoo
from ...learner.spamsfunc import *

def nullcallback(epoch):
	pass
class SparseRUWLearner(object):
	"""
	A region user word learner learns form the loss function:
	Y - wXu where X is a tensor of size:
	
		X.shape == (N,R,U,W)
		Y.shape == (N,T,R)

	where:

		N = number of days of data
		R = number of regions
		U = number of users
		W = number of words
		T = number of tasks

	from the loss definition we learn w and u such that:
		w.shape = (R,T,W)
		u.shape = (R,T,U)

	i.e. a weighting for a word or a user for a task and a region

	regularisation is applied:
		- for users across tasks (users are held independently by region)
		- for words across tasks and regions 

	an multi task l1l2 regulariser is applied for users 
		- few users are selected with some weighting across all tasks

	a graph regulariser is applies for words:
		- few words are selected with some weighting across all tasks and regions
		  for selected words

	if intercept mode is turned on, bias is learnt during the user learning step 
	and subtracted from Y in the word learning step


	"""

	def __init__(self,u_spams,w_spams,**params):
		super(SparseRUWLearner, self).__init__()
		self.params = {
			"epochs": 10,
			"intercept": True,
			"bilinear_tolerance": 0.0001,
			"epoch_callback": nullcallback
		}
		for x,y in params.items(): self.params[x] = y
		self.u_spams = u_spams
		self.w_spams = w_spams
		
		self.epoch_dict = None

		if self.params["intercept"]:
			u_spams.params["intercept"] = True
			w_spams.params["intercept"] = False

	def _reset_epoch(self,epoch,Xu,Xw,Y):
		self.epoch_dict = {}; 
		self.epoch_dict["epoch"] = epoch;
		self.epoch_dict["key_order"] = [];
		self._ed("w_spams_params",self.w_spams.params);
		self._ed("u_spams_params",self.u_spams.params);
		# self._ed("Xu",Xu);
		# self._ed("Xw",Xw);
		# self._ed("Y",Y);

	def _ed(self,k,v):
		if self.params['epoch_callback'] is nullcallback: return v
		if not k in self.epoch_dict['key_order']: self.epoch_dict['key_order']+=[k]
		self.epoch_dict[k] = v
		return v

	def learn(self,Xu,Xw,Y):
		"""
		Expected input: 
			Xu - A (sparse) matrix of shape [R * N * U, W]. 
				 Contains the same data as Xw.
			Xw - A (sparse) matrix of shape [R*N*W,U]
				 Contains the same data as Xu.
			Y  - A tensor of shape  [N, T, R]. 
				 Contains the value of a task on a day in a region

		"""
		self.U = Xw.shape[1] # Xw is of shape 
		self.W = Xu.shape[1]
		self.N = Y .shape[0]
		self.T = Y .shape[1]
		self.R = Y .shape[2]

		# initialise estimates
		u_hat = np.ones((self.R, self.T, self.U))
		w_hat = np.ones((self.R, self.T, self.W))
		b_hat = np.zeros((self.T, self.R))

		old_u_hat = u_hat
		old_w_hat = w_hat


		error = lambda: self._calculate_error(Y,Xw,u_hat,w_hat,b_hat)
		tol = lambda a,b:norm(a.flatten() - b.flatten(),2) < self.params['bilinear_tolerance']
		sparcity = lambda a: (float(sum((a == 0)))/a.size)
		e=sys.float_info.max
		for epoch in range(self.params["epochs"]):
			self._reset_epoch(epoch,Xu,Xw,Y)
			# phase 1: learn u & b given fixed w
			self._ed("u_hat_before",dc(u_hat))
			self._ed("w_hat_before",dc(w_hat))
			self._ed("b_hat_before",dc(b_hat))
			self._ed("error_before",e)
			logger.debug("... Epoch Start Error: %s"%e)
			###### UPDATE W ###########
			w_hat = self._learnW(Y,Xw,u_hat,w_hat,b_hat)
			logger.debug("... w sparcity: %2.2f"%sparcity(w_hat))
			e=error()
			self._ed("error_after_word",e)
			logger.debug("... Error after word: %s"%e)
			###### UPDATE U ###########
			u_hat,b_hat = self._learnU(Y,Xu,u_hat,w_hat,b_hat)
			logger.debug("... u sparcity: %2.2f"%sparcity(u_hat))
			e=error()
			self._ed("error_after_user",e)
			logger.debug("... Error after user: %s"%e)
			

			if tol(u_hat,old_u_hat) and tol(w_hat,old_w_hat):
				logger.debug("No change in previous epoch in either w or u, ending early")
				break
			old_w_hat = w_hat
			old_u_hat = u_hat
			self.params['epoch_callback'](self.epoch_dict)

		return u_hat,w_hat,b_hat
	def _calculate_error(self,Y,Xw,u_hat,w_hat,b_hat):
		# logger.debug("Calculating error...")
		Dstack = self._stack_D(Xw,u_hat)
		Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
		flatw = array([w_hat.flatten()]).T
		Yest = Dstack.dot(flatw)
		# This is:
		# 	- making Yest into something the same shape as Y
		# 	- Then adding the bias
		#	- Then flattening Yest again
		# It is hidious and should not be done like this.
		Yest = array([(Yest.reshape([self.R,self.T,self.N]
				).transpose([2,1,0]) + b_hat
			).transpose([2,1,0]).flatten()]).T
		# uerr_regul = sum([self.u_spams._regul_error(u_hat[x,:,:]) for x in range(u_hat.shape[0]) ])
		uerr_regul = 0
		werr = self.w_spams.error(Yest,Yntrflat,flatw)
		# logger.debug("w_err: %2.2f"%werr)
		# logger.debug("u_err regul: %2.2f"%uerr_regul)
		err =  werr + uerr_regul
		# logger.debug("Done calculating error...")
		return err

	def _stack_V(self,Xu,w_hat):
		# logger.debug("Creating V matrix...")
		start_time = round(time.time() * 1000)
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R
		V = [
			Xu[
				(r*N*U):(r+1)*N*U, :# Grab the r'th block of User/Day rows (r -> r + 1)
			].dot(
				ssp.csc_matrix(
					w_hat[r,:,:] # Dot product with the r'th weighting
				).T
			) 
			for r in range(R)
		]
		end_time = round(time.time() * 1000)
		# logger.debug("Done dot product, tool=%d ..."%(end_time-start_time))
		start_time = round(time.time() * 1000)
		V = [
			ssp.vstack([
					# reshapeflat(V[r][:,t].T,(N,U))
					# V[r][:,t].tocoo().reshape((N,U))
					reshapecoo(V[r][:,t].T,(N,U)) # the fastest way so far
					for t in range(T)
				], format=("csc")
			) 
			for r in range(R)
		]
		end_time = round(time.time() * 1000)
		# logger.debug("Done creating V matrix, tool=%d ..."%(end_time-start_time))
		return V

	def _learnU(self,Y,Xu,u_hat,w_hat,b_hat):
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R
		# phase 1: learn u & b given fixed w
		
		V = self._stack_V(Xu,w_hat)

		# self._ed("V",V)
		# vr_list = self._ed("Vr",[])
		for r in range(R):
			# this is (N x U x T)
			Yr = Y[:,:,r]
			Vrstack = V[r]
			Yrstack = zeros((T*N,T)) + nan
			for t in range(T):
				Yrstack[t*N:(t+1)*N,t:t+1] = Yr[:,t:t+1]
				
			Yspams = asfortranarray(Yrstack)
			Xspams = Vrstack

			ur0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
			# ur0 = ssp.csr_matrix((Xspams.shape[1],Yspams.shape[1]))
			ur,ur_bias = self.u_spams.call(Xspams,Yspams,ur0)
			u_hat[r,:,:] = ur.T
			if self.params["intercept"]:
				b_hat[:,r] = ur_bias
			# epoch_res_r = {}
			# vr_list += [epoch_res_r]
			# epoch_res_r['Xspams'] = dc(Xspams)
			# epoch_res_r['Yspams'] = dc(Yspams)
			# epoch_res_r['ur0'] = dc(ur0)
			# epoch_res_r['params'] = dc(self.u_spams.params)
			# epoch_res_r['ur'] = dc(u_hat[r,:,:])
			# epoch_res_r['br'] = dc(b_hat[:,r])
			
		self._ed("u_hat",dc(u_hat))
		self._ed("b_hat",dc(b_hat))
		return u_hat,b_hat

	def _stack_D(self,Xw,u_hat):
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R
		Dstack = ssp.lil_matrix((N*R*T,W*R*T))

		i = 0;
		NW = N * W
		# logger.debug("Calling _stack_D")
		for r in range(R):
			# I expect this to be (N * W) * U
			Xwr = Xw[r*NW:(r+1)*NW,:]
			for t in range(T):
				# I expect this to be (N * W) * 1
				# Xwrt = Xwr.dot(u_hat[r,t,:])
				Xwrt = ssp.csr_matrix(Xwr.dot(u_hat[r,t,:]))
				for n in range(N):
					DSn = (i * N + n)
					DSw = (i * W )
					# Dstack.rows[DSn] = range(DSw,DSw + W)
					# Dstack.data[DSn] = Xwrt[n*W:n*W + W]
					sub = ssp.lil_matrix(Xwrt[:,n*W:n*W + W])
					Dstack.rows[DSn] = [x + DSw for x in sub.rows[0]]
					Dstack.data[DSn] = sub.data[0]
				i+=1 
		# logger.debug("Done")
		return Dstack

	def _learnW(self,Y,Xw,u_hat,w_hat,b_hat):
		# logger.debug("Learning W")
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R

		Dstack = self._stack_D(Xw,u_hat)
		# logger.debug("... Dstack formed, nnz: %d"%Dstack.nnz)
		Yntrflat = array([ (Y-b_hat).transpose([2,1,0]).flatten()]).T
		Yspams = asfortranarray(Yntrflat)
		Xspams = Dstack.tocsc()
		wr0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
		wr,_ = self.w_spams.call(Xspams,Yspams,wr0)	
		# Dd = self._ed("D",{})
		# Dd['params'] = self.w_spams.params
		# Dd['Xspams'] = Xspams
		# Dd['Yspams'] = Yspams
		# Dd['wr0'] = wr0
		# Dd['wr'] = wr
		w_hat = wr.reshape([R,T,W])
		self._ed("w_hat",dc(w_hat))

		return w_hat

def prep_uspams(**otherargs):
	spamsParams = {
		"loss":"square-missing",
		"compute_gram":False,
		"regul":"l1l2",
		'lambda1' : 0.5, 
	}
	spamsParams = dict(spamsParams.items() + otherargs.items())
	return FistaFlat(**spamsParams)

def prep_w_graphbit(U,W,T,R):
	# set up the group regul
	wrindex = arange(R*T*W).reshape([R,T,W])
	ngroups = W * (T + R)
	eta_g = ones(ngroups,dtype = np.float)
	groups = ssp.csc_matrix(
		(ngroups,ngroups),dtype = np.bool
	)

	groups_var = ssp.dok_matrix((W*R*T,ngroups),dtype=np.bool)
	i = 0
	logger.debug("Creating the sausage groups")
	allgs = []
	for word in range(W):
		for r in range(R):
			allgs += [wrindex[r,:,word]]
			groups_var[wrindex[r,:,word],i] = 1
			i+=1
		for t in range(T):
			groups_var[wrindex[:,t,word],i] = 1
			allgs += [wrindex[:,t,word]]
			i+=1

	logger.debug("Done creating group_var")
	groups_var = ssp.csc_matrix(groups_var,dtype=np.bool)
	graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
	return graph, allgs

def prep_wspams(U,W,T,R, graphbit=None, **otherargs):
	logger.debug("Preparing the graph regulariser")
	graphparams = {
		"loss":"square",
		"regul":"graph",
		'lambda1' : 0.5,
		'verbose' : False, "missing": False
	}
	if not graphbit: graphbit = prep_w_graphbit(U,W,T,R)
	graph,allgs = graphbit
	graphparams = dict(graphparams.items() + otherargs.items())
	return FistaGraph(graph,allgs,**graphparams)