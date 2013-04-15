from bivariate.generator.billmatlabgenerator import *

MATLAB_FILE_LOC="/home/ss/Dropbox/TrendMiner/deliverables/year2-18month/Austrian Data/data.mat"
def test_import():
	gen = BillMatlabGenerator(MATLAB_FILE_LOC,98,True)

	X,Y = gen.generate()

	print X.shape
	print Y.shape