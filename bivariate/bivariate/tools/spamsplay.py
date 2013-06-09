from bivariate.learner.partbatchbivariate import BatchBivariateLearner
import logging;logger = logging.getLogger("root")
from pylab import *
import spams
import scipy.sparse as ssp
import sys

if len(sys.argv[1:]) != 4:
	nusers = 10
	nwords = 20
	ntasks = 1
	ndays = 3
else:
	(nusers, nwords, ntasks, ndays) = [int(x) for x in sys.argv[1:]]
print ""
W = np.asfortranarray(zeros((nwords,ntasks)))
U = ssp.csc_matrix(ones((nusers,ntasks)))
X = ssp.rand(nwords,nusers*ndays,format="csc")
Y = np.asfortranarray(rand(ndays,ntasks))
Y = BatchBivariateLearner._expandY(Y)

logger.debug("Input created!")
def cols_for_day(day):
	return slice(day * nusers, (day+1) * nusers)
logger.debug("Creating Vprime!")
Vprime = BatchBivariateLearner._calculateVprime(X,U)
logger.debug("Calculating W")
W = spams.fistaFlat(Y,Vprime,W,False,loss="square",regul="l1",lambda1=0.01)
W = ssp.csc_matrix(W)
logger.debug("Creating DPrime")
Dprime = BatchBivariateLearner._calculateDprime(X,W,U.shape)
U = np.asfortranarray(zeros((nusers,ntasks)))
logger.debug("Calculating U")
U = spams.fistaFlat(Y,Dprime,U,False,loss="square",regul="l1",lambda1=0.01)
