#!/usr/bin/env python
from bivariate.dataset import userwordregion
from bivariate.dataset import regionpolls as rpt
from scipy import io as sio
from scipy import sparse as ssp
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
		return userwords,False
	except Exception, e:
		logger.debug("scipy load failed, trying h5py load")
	import h5py
	renormd = h5py.File(f)
	alldata = renormd[key]
	data = alldata['data']
	ir = alldata['ir']
	jc = alldata['jc']
	mat = ssp.csc_matrix((data, ir, jc))
	logger.debug("Data loaded, turning into lil")
	return mat.tolil(),True



parser = OptionParser()
parser.add_option("-m", "--user-region-map", dest="user_region_map",
                  help="File containing the user to region man", metavar="FILE",default=USER_REGION_MAP_FILE)
parser.add_option("-w", "--user-word-matrix", dest="user_word_matrix", default=USER_WORD_MATRIX, metavar="FILE",
                  help="Matlab file containing a matrix of size: (user * day) x words")
parser.add_option("-p", "--poll-data", dest="poll_file", default=POLL_DATA, metavar="FILE",
                  help="A matlab file containing region polls as matrices with 'region' in their name")
parser.add_option("-n", "--number-of-days", dest="ndays", default=15,
                  help="Number of days")
parser.add_option("-k", "--user-word-matrix-key", dest="user_word_matrix_key", default="subvar2_vsr",
                  help="The key to get the matrix from in the file")
parser.add_option("-o", "--output", dest="output", default=".",
                  help="Root directory to output the matrices")
parser.add_option("-f", "--force-h5py-output", dest="force_h5py", action="store_true", 
				  default=False,
                  help="Force output in h5py mode")

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

userwords,useh5py = loaduserwords(options.user_word_matrix,options.user_word_matrix_key)
useh5py = useh5py or options.force_h5py
logger.debug("Constructing flat region matrices")
regiondayuserword, regiondayworduser = userwordregion.transform(
	userwords,userregionmap,int(options.ndays)
)

logger.debug("Loading polls")
regionpolls = rpt.transform(options.poll_file)
if not os.path.exists(options.output): os.makedirs(options.output)
X_out = "%s/X.mat"%options.output
meta_out = "%s/meta.mat"%options.output
Y_out = "%s/Y.mat"%options.output

logger.debug("Outputting meta information to: %s"%meta_out)
sio.savemat(
	meta_out,
	{
		"D": options.ndays, 
		"R": regionpolls.shape[2], "T": regionpolls.shape[1],
		"W": regiondayuserword.shape[1],
		"U": regiondayworduser.shape[1]
	}
)
logger.debug("Outputting poll information to: %s"%Y_out)
sio.savemat(
	Y_out,
	{
		"regionpolls":regionpolls
	}
)
logger.debug("Outputting main X matrix to: %s"%X_out)
if useh5py:
	import h5py
	logger.debug("Saving main matrices with h5py")
	def save_group(h5file, name, mat):
		if not ssp.isspmatrix_csr(mat): mat = mat.tocsr()
		
		tosaveg = h5file.create_group(name)
		tosaveg.create_dataset("indptr",data=mat.indptr,compression='lzf')
		tosaveg.create_dataset("data",data=mat.data,compression='lzf')
		tosaveg.create_dataset("indices",data=mat.indices,compression='lzf')

	tosave = h5py.File(X_out, "w")
	logger.debug("Saving userword group")
	save_group(tosave,"regiondayuserword",regiondayuserword)
	logger.debug("Saving worduser group")
	save_group(tosave,"regiondayworduser",regiondayworduser)
	
	tosave.flush()
	tosave.close()
else:
	logger.debug("Saving main matrices with scipy")
	sio.savemat(
		X_out, {
			"regiondayuserword":regiondayuserword, 
			"regiondayworduser":regiondayworduser,
		}
	)