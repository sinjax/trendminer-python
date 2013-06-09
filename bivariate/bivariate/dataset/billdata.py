import h5py
from pylab import *
import scipy.sparse
import os
import scipy.io
import logging;logger = logging.getLogger("root")
from IPython import embed


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
	"""
	def __init__(self, userCol_in, wordCol_in, ndays):
		super(SparseUserDayWord, self).__init__()
		self.ndays = ndays

		if type(userCol_in) is scipy.sparse.csc_matrix:
			logger.debug("Inputs are sparse matricies")
			self.userCol = userCol_in
			self.wordCol = wordCol_in
			self.nwords = self.wordCol.shape[1]
			self.nusers = self.userCol.shape[1] / self.ndays
			self.mode = 0
			return
		if type(userCol_in) is not str:
			raise Exception("The inputs must be either csc_matrix sparse matrix or strings to files containing such matricies")
		logger.debug("Word col file: %s"%os.path.basename(wordCol_in))
		logger.debug("User col file: %s"%os.path.basename(userCol_in))
		logger.debug("Number of days: %d"%ndays)
		try:
			self.userCol = scipy.io.loadmat(userCol_in)
			self.userCol = self.userCol[[x for x in self.userCol.keys() if not x.startswith("_")][0]]
			self.wordCol = scipy.io.loadmat(wordCol_in)
			self.wordCol = self.wordCol[[x for x in self.wordCol.keys() if not x.startswith("_")][0]]
			self.nwords = self.wordCol.shape[1]
			self.nusers = self.userCol.shape[1] / self.ndays
			logger.debug("Using loadmat")
			self.mode = 0
			return
		except Exception, e:
			print e
			self.mode = 1
		logger.debug("Using h5py")
		self.wordColF = h5py.File(wordCol_in).values()[0]
		self.nwords = self.wordColF['jc'].shape[0]-1
		self.userColF = h5py.File(userCol_in).values()[0]
		
		self.nusers = (self.userColF['jc'].shape[0] -1 ) / self.ndays
		


	@classmethod
	def loadCSC(cls,data, indices, indptr,shape=None):
		return scipy.sparse.csc_matrix((data,indices,indptr),shape=shape)

	def list(self,days=None):
		if not days: days = (0,self.ndays)
		logger.debug("Gather a list of days: %s"%str(days))
		alldays = []
		for day in range(days[1]):
			alldays += [self.mat(days=(day,day+1))]
		return alldays

	def mat(self,days=None,word_subsample=None):
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
			if days:
				firstDay,lastDay = extractFirstLast(days)

				logger.debug("Extracting days: %s -> %s"%(firstDay,lastDay))
				expectedCols = self.nusers * (lastDay - firstDay)
				# output must be self.nwords x expectedCols
				outputDims = (self.nwords,expectedCols)
				logger.debug("Expected dimensions: %s"%str(outputDims))
				uptr = array(self.userColF['jc'][firstDay*self.nusers:(lastDay*self.nusers)+1],dtype=int32)
				udta = array(self.userColF['data'][uptr[0]:uptr[-1]],dtype=float64)
				uind = array(self.userColF['ir'][uptr[0]:uptr[-1]],dtype=int32)
				uptr -= uptr[0]
				logger.debug("Creating refined usercol sparse matrix")
				userCol = SparseUserDayWord.loadCSC(udta,uind,uptr,shape=outputDims)
				logger.debug("Creating wordCol matrix (via transpose and csc_matrix)")
				wordCol = scipy.sparse.csc_matrix(userCol.transpose())

			logger.debug("Done creating matrix")
			
		elif self.mode is 0:
			if days is None:
				userCol = self.userCol
				wordCol = self.wordCol
			else:
				firstDay,lastDay = extractFirstLast(days)
				userCol = self.userCol[:,firstDay*self.nusers:lastDay*self.nusers]
				wordCol = scipy.sparse.csc_matrix(userCol.transpose())

		if word_subsample is not None:
			nwords = userCol.shape[0]
			wrds = rand(userCol.shape[0])>(1 - word_subsample)
			wrds = np.nonzero(wrds)[0]
			logger.debug("Load complete, subsampling %d words"%wrds.shape)

			userCol = userCol[wrds,:]
			wordCol = wordCol[:,wrds]
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
		p2 = scipy.io.loadmat(loc);
		self.yvalues = p2[tasks_key][p2[tasks_index]-1,:]
		self.yvalues = self.yvalues.reshape(self.yvalues.shape[0],self.yvalues.shape[2])

	def mat(self,days=None):
		if not days:
			return self.yvalues
		else:
			start,end = days
			return self.yvalues[start:end,:]
		

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
	scipy.io.savemat(loc, {matname : sparsemat})
	
def suserdayword(userCol, wordCol,ndays):
	return SparseUserDayWord(userCol,wordCol,ndays)

def taskvals(loc,**xargs):
	return TasksAcrossDays(loc,**xargs)
