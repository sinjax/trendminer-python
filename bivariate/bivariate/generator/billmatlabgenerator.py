from scipy.io import loadmat

class BillMatlabGenerator(object):
	"""docstring for BillMatlabGenerator"""
	def __init__(self, matlabfile, ndays, filter):
		super(BillMatlabGenerator).__init__()
		self.matlabfile = matlabfile
		self.datadict = loadmat(matfile)
		self.ndays = ndays
		self.currentIndex = 0
		self.filter = filter
		self.prepareVocabulary()
		self.prepareDayPolls()
		self.prepareDayUserWords()

	def prepareVocabulary(self):
		voc = [x[0][0] for x in self.datadict['voc']]
		self.keepindex = [x for x in self.datadict["voc_keep_terms_index"][0]]
		keepindexset = set(keepindex)
		vocIndex=0
		index=0
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
		self.dayPolls = [allDayTasks[d,:] for x in range(self.ndays)]

	def prepareDayUserWords(self):
		arr = self.datadict['user_vsr_for_polls']
		self.nusers = arr.shape[0]/self.ndays
		self.dayWords = [
			arr[
				nusers * d : nusers * (d+1),
				self.keepindex
			].T 
			for d in range(self.ndays)
		]

	def generate(self):
		if self.currentIndex >= self.dayWords:
			return None
		toret = [self.dayWords[self.currentIndex],self.dayPolls[self.currentIndex]]
		self.currentIndex+=1
		return toret


