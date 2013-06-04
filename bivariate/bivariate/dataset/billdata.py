import h5py
from pylab import *
import scipy.sparse
import logging as logger
import logging.config
import os
import scipy.io

logger.config.fileConfig("logconfig.ini")

class SparseUserDayWord(object):
	"""
	This data format loads a sparse matrix containing as rows
	users grouped by days and as columns the values of words.

	This data structure holds both a row centric and column centric
	format of the data for quick access

	either all the data can be loaded or specific days (or sets of days)
	can be loaded very efficiently 

	This dataset must be told the 3 things.
	wordColFile - The file containing the hdf5 compressed matlab sparse 
				  matrix s.t. the columns contain the words
	userColFile - The file containing the hdf5 compressed matlab sparse 
				  matrix s.t. the columns contain the user/days
	ndays - So the number of users can be decerned, the number of days this data
			represents must be told
	"""
	def __init__(self, wordColFile, userColFile, ndays):
		super(SparseUserDayWord, self).__init__()
		logging.debug("Word col file: %s"%os.path.basename(wordColFile))
		logging.debug("User col file: %s"%os.path.basename(userColFile))
		logging.debug("Number of days: %d"%ndays)
		self.wordColF = h5py.File(wordColFile).values()[0]
		self.userColF = h5py.File(userColFile).values()[0]
		
		self.ndays = ndays
		self.nusers = (self.userColF['jc'].shape[0] -1 ) / self.ndays
		self.nwords = self.wordColF['jc'].shape[0]-1


	@classmethod
	def loadCSC(cls,data, indices, indptr,shape=None):
		return scipy.sparse.csc_matrix((data,indices,indptr),shape=shape)
		
	def mat(self,days=None):
		wordCol = None
		userCol = None
		if days is None and users is None and words is None:
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
			firstDay = days[0]
			lastDay = days[1]
			logging.debug("Extracting days: %s -> %s"%(firstDay,lastDay))
			if firstDay < 0: firstDay = 0
			if lastDay > self.userColF['jc'].shape[0]: lastDay = self.userColF['jc'].shape[0]-1
			expectedCols = self.nusers * (lastDay - firstDay)
			# output must be self.nwords x expectedCols
			outputDims = (self.nwords,expectedCols)
			logging.debug("Expected dimensions: %s"%str(outputDims))
			uptr = array(self.userColF['jc'][firstDay*self.nusers:(lastDay*self.nusers)+1],dtype=int32)
			udta = array(self.userColF['data'][uptr[0]:uptr[-1]],dtype=float32)
			uind = array(self.userColF['ir'][uptr[0]:uptr[-1]],dtype=int32)
			uptr -= uptr[0]
			logger.debug("Creating refined usercol sparse matrix")
			userCol = SparseUserDayWord.loadCSC(udta,uind,uptr,shape=outputDims)

			logger.debug("Creating wordCol matrix (via transpose and csc_matrix)")
			wordCol = scipy.sparse.csc_matrix(userCol.transpose())

		logging.debug("Done creating matrix")
		return (userCol,wordCol)


def suserdayword(wordCol,userCol,ndays):
	return SparseUserDayWord(wordCol,userCol,ndays)
if __name__ == '__main__':
	p2 = scipy.io.loadmat("/home/ss/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartII.mat");

	yvalues = p2["polls_vi_cum"][p2["polls_index"]-1,:]
	udw = suserdayword(
		"/home/ss/Experiments/twitter_uk_users_MATLAB/user_vsr_for_polls.mat",
		"/home/ss/Experiments/twitter_uk_users_MATLAB/user_vsr_for_polls_t.mat",
		yvalues.shape[0]
	)
	
	ua,wa = udw.mat(days=(81,81+190))
	ub,wb = udw.mat(days=(81+190,81+190+4),appendTo=(ua,wa))
