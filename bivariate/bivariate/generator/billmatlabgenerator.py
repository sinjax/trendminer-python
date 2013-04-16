from scipy.io import loadmat
from pylab import *
		
class BillMatlabGenerator(object):
	"""docstring for BillMatlabGenerator"""
	def __init__(self, matlabfile, ndays, filter):
		super(BillMatlabGenerator,self).__init__()
		self.matlabfile = matlabfile
		self.datadict = loadmat(matlabfile)
		self.ndays = ndays
		self.currentIndex = 0
		self.filter = filter
		self.prepareVocabulary()
		self.prepareDayPolls()
		self.prepareDayUserWords()
		self.prepareFolds()

	def fromFold(self,inds):
		words = [self.dayWords[i] for i in inds]
		polls = vstack([self.dayPolls[i] for i in inds])

		return (words,polls)

	def prepareFolds(self):
		folds = self.datadict['set_fold']
		self.folds = []
		frommat = lambda mat: [x-1 for x in mat[0,:]]
		for fold in folds:
			foldDict = {
				"training":frommat(fold[0]),
				"test":frommat(fold[1]),
				"validation":frommat(fold[2])

			}
			self.folds += [foldDict]

	def prepareVocabulary(self):
		voc = [x[0][0] for x in self.datadict['voc']]
		self.keepindex = [x-1 for x in self.datadict["voc_keep_terms_index"][0]]
		keepindexset = set(self.keepindex)
		vocIndex=0
		index=0
		self.voc = dict()
		self.indexToVoc = dict()
		for vocArrItem in voc:
			if self.filter and index in keepindexset:
				self.voc[index] = vocArrItem
				self.indexToVoc[index] = vocIndex
				vocIndex+=1
			index+=1

	def prepareDayPolls(self):
		pollKeys = [x for x in self.datadict.keys() if x.endswith("unique_extended")]
		self.ntasks = len(pollKeys)
		self.tasks = [None in range(self.ntasks)]
		allDayTasks = hstack([
			self.datadict[x] for x in pollKeys
		])
		self.dayPolls = [allDayTasks[d:d+1,:] for d in range(self.ndays)]

	def prepareDayUserWords(self):
		arr = self.datadict['user_vsr_for_polls']
		self.nusers = arr.shape[0]/self.ndays
		self.dayWords = [
			arr[
				self.nusers * d : self.nusers * (d+1),
				self.keepindex
			].T 
			for d in range(self.ndays)
		]

	def generate(self):
		if self.currentIndex >= self.dayWords:
			return None
		toret = [
			self.dayWords[self.currentIndex].astype(float32),
			self.dayPolls[self.currentIndex]
		]
		self.currentIndex+=1
		return toret


