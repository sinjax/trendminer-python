import os 
import argparse
from IPython import embed
import datetime
import time
from scipy import sparse as ssp
from scipy import io as sio
from pylab import sqrt,mean
from spectral import spectral

parser = argparse.ArgumentParser(description='Turn a file into a matrix')
def file_exists(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return arg
parser.add_argument(
	"-tweets", dest="tweets",help="File containing tweets in JSON, one per line", metavar="FILE", required=True, 
	type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-npmi", dest="npmi",help="File containing NPMI of words", metavar="FILE", required=True, 
	type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-dict", dest="vdict",help="File containing the words", metavar="FILE", required=True, 
	type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-k", dest="k",help="K for the KNN",  required=False, 
	default=30
)

parser.add_argument(
	"-c", dest="c",help="Number of clusters in spectral clustering",  required=False, 
	default=30
)
options = parser.parse_args()
spectral(options.tweets,options.npmi,options.vdict,int(options.k),int(options.c))
