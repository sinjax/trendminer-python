from pylab import *
import scipy.io as sio
import scipy.sparse as ssp
import argparse
from IPython import embed

def read_dayuserwords(loc):
	f = file(loc,"r")
	for line in f.readlines():
		embed()

parser = argparse.ArgumentParser(description='Load text summaries as matrices')
parser.add_argument("-duw", dest="duw",help="File containing day user word counts", metavar="FILE",required=True)
parser.add_argument("-dut", dest="dut",help="File containing day user document counts", metavar="FILE",required=True)

options = parser.parse_args()

day_user_words = read_dayuserwords(options.duw)
day_user_total = read_dayusertotal(options.dut)

