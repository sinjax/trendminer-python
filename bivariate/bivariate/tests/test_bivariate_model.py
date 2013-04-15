from bivariate.learner.batchbivariate import BatchBivariateLearner
from bivariate.generator.billmatlabgenerator import *


MATLAB_FILE_LOC="/home/ss/Dropbox/TrendMiner/deliverables/year2-18month/Austrian Data/data.mat"

def test_bivariate_model():
	spamsDict = {
		"numThreads": 1, "verbose": False,
		"lambda1": 0.1, "it0": 3, "max_it": 10,
		"L0":0.1,"tol":1e-3,"intercept" : False, "pos": False
	}
	spamsDict['compute_gram'] = True
	spamsDict["loss"] = "square"
	spamsDict["regul"] = "l1l2"
	learner = BatchBivariateLearner(**spamsDict)
	gen = BillMatlabGenerator(MATLAB_FILE_LOC,98,True)

	for i in range(10):
		X,Y = gen.generate()
		learner.process(X,Y)