from bivariate.crossvalidation import timeseriescv 
from pylab import *

def test_tsfi():
	tsfi = timeseriescv.tsfi(20,ntraining=10,ntest=2,nvalidation=2)
	for x in tsfi:
		assert len(x.training) >= 10
		assert len(x.test) == 2
		assert len(x.validation) == 2
	tsfi = timeseriescv.tsfi(20,ntraining=5,ntest=3,nvalidation=3)
	for x in tsfi:
		assert len(x.training) >= 5
		assert len(x.test) == 3
		assert len(x.validation) == 3
