import h5py
from pylab import *
import scipy.sparse as ssp
import os
import scipy.io as sio
import logging;logger = logging.getLogger("root")
from IPython import embed

def count_cols_h5(f):
	f = h5py.File(f).values()[0]
	return f['jc'].shape[0]-1

class SparseUserDayWord(object):
	"""
	This data format loads a sparse matrix containing as rows
	users grouped by days and as columns the values of words.

	This data structure holds both a row centric and column centric
	format of the data for quick access

	either all the data can be loaded or specific days (or sets of days)
	can be loaded very efficiently 

	This dataset must be told the 3 things.
	wordCol_in - the words held in the column
				 sparse matrix or loadmat-able sparse matrix or hdf5 sparse matrix from matlab
	userCol_in - the user/days held in the column
				 sparse matrix or loadmat-able sparse matrix or hdf5 sparse matrix from matlab
	ndays - So the number of users can be decerned, the number of days this data
			represents must be told
	nwords - So the sparse matricies don't have to be loaded in their entierty the number of words is needed
	"""
	def __init__(self, userCol_in, wordCol_in, ndays,nwords=None):
		super(SparseUserDayWord, self).__init__()
		self.ndays = ndays
		self.loadWordCol = False
		if wordCol_in is None:
			self.generateWordCol = False
		else:
			self.generateWordCol = True
			if os.path.exists(wordCol_in):
				self.loadWordCol = True
		if type(userCol_in) is ssp.csc_matrix:
			logger.debug("Inputs are sparse matricies")
			self.userCol = userCol_in
			if self.generateWordCol and type(wordCol_in) is ssp.csc_matrix: 
				self.wordCol = wordCol_in
			self.nwords = self.userCol.shape[0]
			self.nusers = self.userCol.shape[1] / self.ndays
			self.mode = 0
			return
		if type(userCol_in) is not str:
			raise Exception("The inputs must be either csc_matrix sparse matrix or strings to files containing such matricies")
		if self.loadWordCol:
			logger.debug("Word col file: %s"%os.path.basename(wordCol_in))
		logger.debug("User col file: %s"%os.path.basename(userCol_in))
		logger.debug("Number of days: %d"%ndays)
		try:
			self.userCol = sio.loadmat(userCol_in)
			self.userCol = self.userCol[[x for x in self.userCol.keys() if not x.startswith("_")][0]]
			if self.loadWordCol:
				self.wordCol = sio.loadmat(wordCol_in)
				self.wordCol = self.wordCol[[x for x in self.wordCol.keys() if not x.startswith("_")][0]]
			self.nwords = self.userCol.shape[0]
			self.nusers = self.userCol.shape[1] / self.ndays
			logger.debug("Using loadmat")
			self.mode = 0
			return
		except Exception, e:
			self.mode = 1
		logger.debug("Using h5py")
		if self.loadWordCol:
			self.wordColF = h5py.File(wordCol_in).values()[0]
			self.nwords = self.wordColF['jc'].shape[0]-1
		else:
			if nwords is None: raise Exception("If the word file is not provided, then the number of words must be provided")
			self.nwords = nwords
		self.userColF = h5py.File(userCol_in).values()[0]
		
		self.nusers = (self.userColF['jc'].shape[0] -1 ) / self.ndays
		

	@classmethod
	def loadCSC(cls,data, indices, indptr,shape=None):
		return ssp.csc_matrix((data,indices,indptr),shape=shape)

	def list(self,days=None):
		if not days: days = (0,self.ndays)
		logger.debug("Gather a list of days: %s"%str(days))
		alldays = []
		for day in range(days[1]):
			alldays += [self.mat(days=(day,day+1))]
		return alldays

	def mat(self,days=None,word_subsample=None,voc=None):
		def extractFirstLast(days):
			firstDay = days[0]
			lastDay = days[1]
			return firstDay,lastDay
		wordCol = None
		userCol = None
		if self.mode is 1:
			wordCol = None
			userCol = None
			if days is None:
				# Load everything
				if self.loadWordCol:
					logger.debug("Loading entire word column matrix")
					wordCol = SparseUserDayWord.loadCSC(
						array(self.wordColF['data'],dtype=float32),
						array(self.wordColF['ir'],dtype=int32),
						array(self.wordColF['jc'],dtype=int32)
					)
				logger.debug("Loading entire user column matrix")
				userCol = SparseUserDayWord.loadCSC(
					array(self.userColF['data'],dtype=float32),
					array(self.userColF['ir'],dtype=int32),
					array(self.userColF['jc'],dtype=int32)
				)
				if self.generateWordCol:
					wordCol = ssp.csc_matrix(userCol.transpose())
			if days:
				firstDay,lastDay = extractFirstLast(days)

				logger.debug("Extracting days: %s -> %s"%(firstDay,lastDay))
				expectedCols = self.nusers * (lastDay - firstDay)
				# output must be self.nwords x expectedCols
				outputDims = (self.nwords,expectedCols)
				logger.debug("Expected dimensions: %s"%str(outputDims))
				uptr = array(self.userColF['jc'][firstDay*self.nusers:(lastDay*self.nusers)+1],dtype=int32)
				uind = array(self.userColF['ir'][uptr[0]:uptr[-1]],dtype=int32)
				udta = array(self.userColF['data'][uptr[0]:uptr[-1]],dtype=float64)
				uptr -= uptr[0]
				logger.debug("Creating refined usercol sparse matrix")
				userCol = SparseUserDayWord.loadCSC(udta,uind,uptr,shape=outputDims)
				if self.generateWordCol:
					logger.debug("Creating wordCol matrix (via transpose and csc_matrix)")
					wordCol = ssp.csc_matrix(userCol.transpose())

			logger.debug("Done creating matrix")
			
		elif self.mode is 0:
			if days is None:
				userCol = self.userCol
				if self.generateWordCol: wordCol = self.wordCol
			else:
				firstDay,lastDay = extractFirstLast(days)
				userCol = self.userCol[:,firstDay*self.nusers:lastDay*self.nusers]
				if self.generateWordCol: wordCol = ssp.csc_matrix(userCol.transpose())

		if voc is not None:
			logger.debug("Correcting vocabulary with provided voc")
			logger.debug("Turning usercol into a row matrix briefly")
			# userCol = userCol.tocsr()[voc,:].tocsc()
			userCol = userCol[voc,:]
			# logger.debug("Selecting voc words")
			# userCol = userCol[voc,:]
			# logger.debug("Turning user col back to column matrix")
			# userCol = ssp.csc_matrix(userCol)
			if self.generateWordCol: wordCol = wordCol[:,voc]
		
		return userCol,wordCol

class TasksAcrossDays(object):
	"""
	Load bill's tasks across days format.
	This is a matlab file which contains some values and some
	valid indexes.

	by default the polls_vi_cum value is indexes by polls_index
	"""
	def __init__(self, loc, tasks_key="polls_vi_cum", tasks_index="polls_index"):
		super(TasksAcrossDays, self).__init__()
		p2 = sio.loadmat(loc);
		self.yvalues = p2[tasks_key][p2[tasks_index]-1,:]
		self.yvalues = self.yvalues.reshape(self.yvalues.shape[0],self.yvalues.shape[2])

	def mat(self,days=None):
		if not days:
			return self.yvalues
		else:
			start,end = days
			return self.yvalues[start:end,:]

class SpamsTree(object):
	"""
	Load bill's version of the carnegy mellon vocabulary tree format

	"""
	def __init__(self, treefile):
		super(SpamsTree, self).__init__()
		logger.debug("Loading spams tree")
		tree = sio.loadmat(treefile,squeeze_me=False,struct_as_record=False)
		self.spamsin = tree['inputSPAMS'][0,0]

	def spamsobj(self):
		tree = {
			'eta_g': np.array(self.spamsin.nodeWeights.flatten(),dtype=np.float64),
			'groups' : ssp.csc_matrix(self.spamsin.parentNodes,dtype=bool),
			'own_variables' :np.array(self.spamsin.firstNodeVars.flatten()-1,dtype=np.int32),
			'N_own_variables' : np.array(self.spamsin.nodeOwnVarsNum.flatten(),dtype=np.int32)
		}
		return tree

class Vocabulary(object):
	"""Some error occurred with the vocabulary, this corrects it"""
	def __init__(self, vocloc):
		super(Vocabulary, self).__init__()
		logger.debug("Loading corrected vocabulary")
		voc_matching = sio.loadmat(vocloc,squeeze_me=False,struct_as_record=False)
		self.voc_index = voc_matching['vocFilteredIndex'][voc_matching['vocPointers_new']]-1
	def voc(self):
		return self.voc_index.flatten()
		
		

def savesparse(sparsemat, loc, matname="mat"):
	if os.path.exists(loc):
		os.remove(loc)
	logger.debug("Saving sparse matrix %s"%loc)
	# f = h5py.File(loc,driver="sec2")
	# g = f.create_group(matname)
	# print sparsemat.data.shape
	# data = g.create_dataset("data",sparsemat.data,compression="gzip",compression_opts=3L,chunks=True)
	# ir = g.create_dataset("ir",sparsemat.indices,compression="gzip",compression_opts=3L)
	# jc = g.create_dataset("jc",sparsemat.indptr,compression="gzip",compression_opts=3L)
	# f.close()
	sio.savemat(loc, {matname : sparsemat})
	
def suserdayword(userCol,ndays, wordCol=None,nwords=None):
	return SparseUserDayWord(userCol,wordCol,ndays,nwords=nwords)

def taskvals(loc,**xargs):
	return TasksAcrossDays(loc,**xargs)

def voc(loc):
	return Vocabulary(loc)
def tree(loc):
	return SpamsTree(loc)

def subsample(userCol,word_subsample=0.001,user_subsample=0.001,ndays=None):

	nwords = userCol.shape[0]
	wrds = rand(nwords)>(1 - word_subsample)
	wrds = np.nonzero(wrds)[0]
	logger.debug("Load complete, subsampling %d words"%wrds.shape)
	

	nusers = userCol.shape[1]/ndays
	usrs = rand(nusers)>(1 - user_subsample)
	usrs = np.nonzero(usrs.flatten())[0]
	usrdays = []
	for x in range(ndays):
		usrdays += (usrs+(x * nusers)).tolist()
	userCol = userCol[wrds,:]
	userCol = userCol[:,usrdays]

	return userCol