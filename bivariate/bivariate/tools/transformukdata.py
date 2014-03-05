#!/usr/bin/env python
from bivariate.dataset import userwordregion
from bivariate.dataset import regionpolls as rpt
from scipy import io as sio
from IPython import embed
import os
import logging;logger = logging.getLogger("root")
USER_REGION_MAP_FILE = "/home/ss/Experiments/bilinear/user-region-map"
USER_WORD_MATRIX = "/home/ss/Experiments/bilinear/user_vsr_polls_15days.mat"
POLL_DATA = "/home/ss/Experiments/bilinear/polls_demographics.mat"
from optparse import OptionParser

def loaduserwords(f,key):
	try:
		logger.debug("Trying to use scipy to load matrix")
		userwordmat = sio.loadmat(f)
		userwords = userwordmat[key]
		return userwords
	except Exception, e:
		logger.debug("scipy load failed, trying h5py load")
	import h5py
	renormd = h5py.File(f)
	alldata = renormd[key]
	data = alldata['data']
	ir = alldata['ir']
	jc = alldata['jc']
	return scipy.sparse.csc_matrix((data, ir, jc))



parser = OptionParser()
parser.add_option("-m", "--user-region-map", dest="user_region_map",
                  help="File containing the user to region man", metavar="FILE",default=USER_REGION_MAP_FILE)
parser.add_option("-w", "--user-word-matrix", dest="user_word_matrix", default=USER_WORD_MATRIX,
                  help="Matlab file containing a matrix of size: (user * day) x words")
parser.add_option("-p", "--poll-data", dest="poll_file", default=POLL_DATA,
                  help="A matlab file containing region polls as matrices with 'region' in their name")
parser.add_option("-n", "--number-of-days", dest="ndays", default=15,
                  help="Number of days")
parser.add_option("-k", "--user-word-matrix-key", dest="user_word_matrix_key", default="subvar2_vsr",
                  help="The key to get the matrix from in the file")
parser.add_option("-o", "--output", dest="output", default=".",
                  help="Root directory to output the matrices")

(options, args) = parser.parse_args()

if not (os.path.exists(options.user_region_map) and os.path.exists(options.user_word_matrix) and os.path.exists(options.poll_file)):
	raise Exception("File's not found!")


logger.debug("Reading map file: %s"%options.user_region_map)

userregion = [
	x.strip().split() 
	for x in file(options.user_region_map).readlines()
]
userregionmap = dict([(int(a[0]), int(a[1])) for a in userregion])

logger.debug("Loading day/user/word matrix: %s"%options.user_word_matrix)

userwords = loaduserwords(options.user_word_matrix,options.user_word_matrix_key)
logger.debug("Constructing flat region matrices")
regiondayuserword, regiondayworduser = userwordregion.transform(
	userwords,userregionmap,options.ndays
)

logger.debug("Loading polls")
regionpolls = rpt.transform(options.poll_file)
if not os.path.exists(options.output): os.makedirs(options.output)
file_out = "%s/XY.mat"%options.output
logger.debug("Outputting file: %s"%file_out)
sio.savemat(
	file_out,
	{
		"D": options.ndays, 
		"R": regionpolls.shape[2], "T": regionpolls.shape[1],
		"W": regiondayuserword.shape[1],
		"U": regiondayworduser.shape[1],
		"regiondayuserword":regiondayuserword, 
		"regiondayworduser":regiondayworduser,
		"regionpolls":regionpolls
	}
)