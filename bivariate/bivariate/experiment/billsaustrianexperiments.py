import time
import logging as logger
import os
from bivariate.learner.batchbivariate import BatchBivariateLearner
from bivariate.generator.billmatlabgenerator import *
from bivariate.evaluator.rootmeaneval import *

DATA_ROOT="/home/ss/Dropbox/TrendMiner/deliverables/year2-18month"\
					+"/Austrian Data"
EXPERIMENTS_ROOT=os.sep.join([DATA_ROOT,"batchExperiments"])
EXPERIMENT_NAME="batch_%d"%(time.time()*1000)
EXPERIMENT_ROOT=os.sep.join([EXPERIMENTS_ROOT,EXPERIMENT_NAME])
MATLAB_DATA=os.sep.join([DATA_ROOT,"data.mat"])

NSTEPS = 20

def prepareExperiment():
	os.makedirs(EXPERIMENT_ROOT)

def runExperiment():
	spamsDict = {
		"numThreads": 1,
		"L0": 0.1,
		"lambda1": 0.001,
		"max_it":500,
		"tol":1e-3,
		"intercept":True, 
		"bivar_it0": 3, 
		"bivar_max_it": 10,
	}
	spamsDict['compute_gram'] = True
	spamsDict["loss"] = "square"
	spamsDict["regul"] = "l1l2"
	learner = BatchBivariateLearner(**spamsDict)
	gen = BillMatlabGenerator(MATLAB_DATA,98,True)
	evaluator = RootMeanSumSquareEval(learner)
	foldN = 0;
	for fold in gen.folds:
		logger.info("Performing fold: %d"%foldN)
		X,Y = gen.fromFold(fold['training'])
		learner.process(X,Y)
		if learner.w is not None and learner.u is not None:
			Xtest,Ytest = gen.fromFold(fold['test'])
			
			logger.info("W sparcity: %2.2f"%learner._wsparcity())
			logger.info("U sparcity: %2.2f"%learner._usparcity())
			logger.info("Bias: %s"%learner.bias)
			loss = evaluator.evaluate(Xtest,Ytest)
			logger.info("Loss: %2.5f"%loss)
			logger.info("The predictions:")
			for i in range(len(Xtest)):
				for t in range(Y.shape[1]):
					dotproduct = learner.u[:,t:t+1].T.dot(Xtest[i].T).dot(learner.w[:,t:t+1])[0,0]
					withoutbias = dotproduct
					if learner.bias is not None: dotproduct += learner.bias[0,t]
					logger.info("task=%d,i=%d,y=%2.5f,v=%2.5f,vb=%2.5f,delta=%3.5f"%(
						t,i,Ytest[i,t],dotproduct,withoutbias,
						pow(Ytest[i,t]-dotproduct,2)
					))
			# calculate loss of Y for new X
			pass
		foldN +=1
		break

if __name__ == '__main__':
	runExperiment()