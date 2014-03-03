import spams
import logging;logger = logging.getLogger("root")
from pylab import *
import scipy.sparse as ssp
import time
from IPython import embed
from ...tools.utils import reshapeflat, reshapecoo


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


	"""
	def __init__(self,u_spams,w_spams):
		super(SparseRUWLearner, self).__init__()
		self.params = {
			"epochs": 10
		}
		self.u_spams = u_spams
		self.w_spams = w_spams


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

		u = np.random.random((self.R, self.T, self.U))
		w = np.random.random((self.R, self.T, self.W))
		b = np.random.random((self.T, self.R))

		# initialise estimates
		u_hat = np.ones_like(u)
		w_hat = np.ones_like(w)
		b_hat = np.ones_like(b)
		error = lambda: self._calculate_error(Y,Xw,u_hat,w_hat)
		for epoch in range(self.params["epochs"]):
			logger.debug("Starting epoch: %d"%epoch)
			# phase 1: learn u & b given fixed w
			logger.debug("Error before user: %s"%error())
			u_hat = self._learnU(Y,Xu,u_hat,w_hat)
			logger.debug("Error after user: %s"%error())
			w_hat = self._learnW(Y,Xw,u_hat,w_hat)
			logger.debug("Error after word: %s"%error())

		return u_hat,w_hat
	def _calculate_error(self,Y,Xw,u_hat,w_hat):
		Dstack = self._stack_D(Xw,u_hat)
		Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
		flatw = array([w_hat.flatten()]).T
		Yest = Dstack.dot(flatw)
		err = self.w_spams.error(Yest,Yntrflat,flatw)
		return err

	def _learnU(self,Y,Xu,u_hat,w_hat):
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R
		# phase 1: learn u & b given fixed w
		logger.debug("Creating V matrix...")
		start_time = round(time.time() * 1000)
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
		logger.debug("Done dot product, tool=%d ..."%(end_time-start_time))
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
		logger.debug("Done creating V matrix, tool=%d ..."%(end_time-start_time))

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
			ur,_ = self.u_spams.call(Xspams,Yspams,ur0)
			u_hat[r,:,:] = ur.T
		return u_hat
	def _stack_D(self,Xw,u_hat):
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R
		Dstack = ssp.lil_matrix((N*R*T,W*R*T))

		i = 0;
		NW = N * W
		for r in range(R):
			# I expect this to be (N * W) * U
			Xwr = Xw[r*NW:(r+1)*NW,:]
			for t in range(T):
				# I expect this to be (N * W) * 1
				Xwrt = Xwr.dot(u_hat[r,t,:])
				for n in range(N):
					DSn = (i * N + n)
					DSw = (i * W )
					Dstack.rows[DSn] = range(DSw,DSw + W)
					Dstack.data[DSn] = Xwrt[n*W:n*W + W]
				i+=1 
		return Dstack
	def _learnW(self,Y,Xw,u_hat,w_hat):
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R

		Dstack = self._stack_D(Xw,u_hat)
		Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
		Yspams = asfortranarray(Yntrflat)
		Xspams = Dstack.tocsc()
		wr0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
		
		wr,_ = self.w_spams.call(Xspams,Yspams,wr0)	
		w_hat = wr.reshape([R,T,W])

		return w_hat
