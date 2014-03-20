from regionuserwordlearner import *
def nullcallback(epoch):
	pass
class SparseRUWMTLearner(SparseRUWLearner):
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

	a multi task graph regulariser is applies for words:
		- few words are selected with some weighting across all tasks and regions
		  for selected words

	if intercept mode is turned on, bias is learnt during the user learning step 
	and subtracted from Y in the word learning step


	"""

	def __init__(self,u_spams,w_spams,**params):
		super(SparseRUWMTLearner, self).__init__(u_spams,w_spams,**params)

	def _calculate_error(self,Y,Xw,u_hat,w_hat,b_hat):
		# logger.debug("Calculating error...")
		Dstack = self._stack_D(Xw,u_hat)
		Yntrflat = array([ Y.transpose([2,1,0]).flatten()]).T
		Ytaskflat = ones((N*R*T,T))
		Ytaskflat[:,:] = NaN
		for t in range(T):
			Ytaskflat[t * N * R:(t+1) * N * R,t] = Yntrflat[t * N * R:(t+1) * N * R][:,0]
asddsaasddas this right here needs fixing!!
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
	def _stack_D(self,Xw,u_hat):
		W,N,T,R = self.W,self.N,self.T,self.R
		return stack_D(Xw,u_hat,W,N,T,R)

	def _learnW(self,Y,Xw,u_hat,w_hat,b_hat):
		logger.debug("Learning W MT")
		U,W,N,T,R = self.U,self.W,self.N,self.T,self.R

		Dstack = self._stack_D(Xw,u_hat)
		Yntrflat = array([ (Y-b_hat).transpose([2,1,0]).flatten()]).T
		Ytaskflat = ones((N*R*T,T))
		Ytaskflat[:,:] = NaN
		for t in range(T):
			Ytaskflat[t * N * R:(t+1) * N * R,t] = Yntrflat[t * N * R:(t+1) * N * R][:,0]
		Yspams = asfortranarray(Ytaskflat)
		Xspams = Dstack.tocsc()
		wr0 = asfortranarray(zeros((Xspams.shape[1],Yspams.shape[1])))
		wr,_ = self.w_spams.call(Xspams,Yspams,wr0)	
		w_hat = wr.reshape([R,W,T]).transpose([0,2,1])
		embed()
		self._ed("w_hat",dc(w_hat))

		return w_hat

def stack_D(Xw,u_hat,W,N,T,R):
	logger.debug("Stacking D MT mode")
	Dstack = ssp.lil_matrix((N*R*T,W*R))

	i = 0;
	NW = N * W
	# logger.debug("Calling _stack_D")
	for t in range(T):
		for r in range(R):
			# I expect this to be (N * W) * U
			Xwr = Xw[r*NW:(r+1)*NW,:]
			# I expect this to be (N * W) * 1
			Xwrt = ssp.csr_matrix(Xwr.dot(u_hat[r,t,:]))
			# epoch_res["Xwrt_r%d_t%d"%(r,t)] = dc(Xwrt)
			for n in range(N):
				DSn = (i * N + n)
				DSw = (r * W )
				sub = ssp.lil_matrix(Xwrt[:,n*W:n*W + W])
				Dstack.rows[DSn] = [x + DSw for x in sub.rows[0]]
				Dstack.data[DSn] = sub.data[0]
			i+=1
	return Dstack
	pass

def prep_w_graphbit(W,T,R):
	# set up the group regul
	wrindex = arange(R*W).reshape([R,W])
	ngroups = W * R
	eta_g = ones(ngroups,dtype = np.float)
	groups = ssp.csc_matrix(
		(ngroups,ngroups),dtype = np.bool
	)

	groups_var = ssp.dok_matrix((W*R,ngroups),dtype=np.bool)
	i = 0
	logger.debug("Creating the sausage groups MT")
	allgs = []
	for r in range(R):
		for word in range(W):
			groups_var[wrindex[r,word],i] = 1
			allgs += [wrindex[r,word]]
			i+=1

	logger.debug("Done creating group_var")
	groups_var = ssp.csc_matrix(groups_var,dtype=np.bool)
	graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
	embed()
	return graph, allgs

def prep_wspams(W,T,R, graphbit=None, **otherargs):
	logger.debug("Preparing the graph regulariser")
	graphparams = {
		"loss":"square",
		"regul":"multi-task-graph",
		'lambda1' : 0.5,
		'lambda2' : 0.5,
		'verbose' : False
	}
	if not graphbit: graphbit = prep_w_graphbit(W,T,R)
	graph,allgs = graphbit
	graphparams = dict(graphparams.items() + otherargs.items())
	return FistaGraph(graph,allgs,**graphparams)
