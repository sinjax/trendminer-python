from IPython import embed
from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from bivariate.generator.randombigenerator import *
from bivariate.learner.spamsfunc import *
from pylab import *
import scipy.sparse as ssp
import bivariate.crossvalidation.timeseriescv as tscv


MATLAB_FILE_LOC="/Users/ss/Dropbox/TrendMiner/deliverables/year2-18month/Austrian Data/data.mat"

def test_bivariate_model():
	spamsDict = {
		"lambda1": 0.01,
		"intercept": True
	}
	spamsDict["loss"] = "square"
	spamsDict["regul"] = "l1"
	w_spams = FistaFlat(**spamsDict)
	u_spams = FistaFlat(**spamsDict)
	learner = BatchBivariateLearner(w_spams,u_spams)
	gen = RandomBiGen()
	ndays = 4
	Xt,Y = gen.generate(n=ndays)
	Xt = vstack(Xt)
	Y = vstack(Y)
	learner.process(Y,Xt=ssp.csc_matrix(Xt))

def test_bivariate_optimise():
	w_spams = FistaFlat(**{
		"intercept": True,
		"loss":"square",
		"regul":"l1"
	})
	u_spams = FistaFlat(**{
		"intercept": True,
		"loss":"square",
		"regul":"l1"
	})
	learner = BatchBivariateLearner(w_spams,u_spams)
	gen = RandomBiGen()
	ndays = 20
	Xt,Y = gen.generate(n=ndays)
	Xt = vstack(Xt)
	Y = vstack(Y)

	folds = [x for x in tscv.tsfi(ndays,ntest=2)]
	lrng = np.arange(0.1,1,0.1)
	fold = folds[0]
	
	X = ssp.csc_matrix(Xt.transpose())

	Xparts,Yparts = BatchBivariateLearner.XYparts(fold,X,Y)
	learner.optimise_lambda(lrng,lrng,Yparts,Xparts)
	
	print w_spams.params
	print u_spams.params