from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from bivariate.generator.randombigenerator import *
from bivariate.learner.spamsfunc import *


MATLAB_FILE_LOC="/home/ss/Dropbox/TrendMiner/deliverables/year2-18month/Austrian Data/data.mat"

def test_bivariate_model():
	w_func = FistaFlat(**{
		"intercept": True,
		"loss":"square",
		"regul":"l1l2",
		"it0":50,
		"max_it":1000,
		"verbose":True
	})
	u_func = FistaFlat(**{
		"intercept": True,
		"loss":"square",
		"regul":"l1l2",
		"it0":50,
		"max_it":1000,
		"verbose":True
	})
	learner = BatchBivariateLearner(w_func,u_func,**spamsDict)
	gen = RandomBiGen(noise=0.01)

	allX = []
	allY = []

	for i in range(1000):
		X,Y = gen.generate()
		allX += [X]
		allY += [Y]

	X = hstack(allX)
	Y = vstack(allY)
	# learner.process(Y,X)
	print X.shape
	print Y.shape