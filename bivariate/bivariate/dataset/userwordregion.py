import logging;logger = logging.getLogger("root")
from scipy import sparse as ssp
from IPython import embed
from pylab import array,hstack,ones
from bivariate.tools.utils import h5py_sparse_row, h5py_save_sparse_row
import shutil
import os
from scipy import io as sio

def for_days(userword,worduser,*days):
	embed()

def read_split_userwordregion(inp,*days,**xargs):
	cache = None; vkeep = None
	if "cache" in xargs:  cache = xargs["cache"]
	if not os.path.exists(inp): raise Exception("Input does not exist")
	if "voc_keep" in xargs: vkeep = xargs['voc_keep']

	R = len(os.listdir(inp))
	allmats = {}
	allmats["worduser"] = []
	allmats["userword"] = []
	for r in range(R):
		rdir = os.sep.join([inp,"r%d"%r])
		for n in days:
			cache_name = "r%dn%d"%(r,n)
			if cache is not None:
				if cache_name in cache:
					for k,mat in cache[cache_name].items():
						allmats[k] += [mat]
					continue
				else:
					cache[cache_name] = dict()
			toload = os.sep.join([rdir,"n%d.mat"%n])
			mats = sio.loadmat(toload)
			wu = mats['worduser']
			uw = mats['userword']
			if vkeep is not None:
				wu = mats['worduser'][vkeep,:]
				uw = mats['userword'][:,vkeep]
			if cache is not None:
				cache[cache_name]["worduser"] = wu
				cache[cache_name]["userword"] = uw

			allmats['worduser'] += [wu]
			allmats['userword'] += [uw]
			if(len(allmats['worduser']) > 1):
				if not allmats['worduser'][-1].shape[0] == allmats['worduser'][-2].shape[0]:
					logger.debug("Incompatible matrix sizes due to different vocabulary size")
					raise Exception("Incompatible matrix sizes due to different vocabulary size")
	
	logger.debug("Stacking word/users")
	rdworduser = ssp.vstack(allmats['worduser'],format="csr")
	logger.debug("Stacking user/words")
	rduserword = ssp.vstack(allmats['userword'],format="csr")

	return rduserword,rdworduser

def split_userwordregion(X,ndays,output):
	if os.path.exists(output):
		shutil.rmtree(output)
	os.makedirs(output)
	userword = X['regiondayuserword']
	worduser = X['regiondayworduser']
	U = worduser.shape[1]
	W = userword.shape[1]
	N = ndays
	R = userword.shape[0] / float(N) / U

	for r in range(R):
		regiondir = os.sep.join([output,"r%d"%r])
		os.makedirs(regiondir)
		for n in range(N):
			dayf = os.sep.join([regiondir,"n%d.mat"%n])
			u_offset = n * U + r * N * U
			urows = [x + u_offset for x in range(U)]
			w_offset = n * W + r * N * W
			wrows = [x + w_offset for x in range(W)]
			logger.debug("Extracting and saving region: %d, day: %d"%(r,n))
			subX = {
				"userword": userword[urows,:],
				"worduser": worduser[wrows,:],
			}
			sio.savemat(dayf, subX)


def transform(dayuserwords, userregionmap, ndays):
	"""
		userwords - a matrix containing a user per row grouped by dayuserwords
		userregionmap - a dictionary of user index to region
		ndays - number of days in userwords

		returns:
			regiondayuserword - sparse matrix containing users grouped by days grouped by region
			regiondayworduser - sparse matrix containing words grouped by days grouped by region
	"""
	regionusermap = dict([(x-1,[]) for x in set(userregionmap.values())])
	for user,region in userregionmap.items():
		regionusermap[region-1] += [user]

	N = ndays
	U = dayuserwords.shape[0]/N
	W = dayuserwords.shape[1]
	R = len(regionusermap)
	missing_users = array(list(set(range(U)) - set(userregionmap.keys())))

	logger.debug("Preparing Output Matrices")
	regiondayuserword = None
	regiondayworduser = None

	# dayuserwords_r = ssp.csr_matrix(dayuserwords)
	if not ssp.isspmatrix_lil(dayuserwords):
		logger.debug("The data array must be lil, transforming...")
		dayuserwords = dayuserwords.tolil()

	
	logger.debug("Filling (R x D x U, W) matrix")
	rows = []
	data = []
	for r in range(R):
		logger.debug("Starting region: %d"%r)
		rusers = set(regionusermap[r])
		for n in range(N):
			logger.debug("Starting day: %d"%n)

			for u in range(U):
				if u not in rusers: 
					rows += [[]]
					data += [[]]
				else:
					i = n * U + u
					rows += [dayuserwords.rows[i]]
					data += [dayuserwords.data[i]]
	
	regiondayuserword = ssp.lil_matrix((1,1),dtype=dayuserwords.dtype)
	regiondayuserword.data = data
	regiondayuserword.rows = rows
	regiondayuserword._shape = (R * N * U, W)
	logger.debug("... cleaning up Filling (R x D x U, W) matrix")
	regiondayuserword = ssp.csr_matrix(regiondayuserword)
	
	logger.debug("Filling (R x D x W, U) matrix")
	rduw_x = regiondayuserword[:U,:]
	regiondayworduser = ssp.coo_matrix(rduw_x.T)
	for x in xrange(1, R * N):
		rduw_x = regiondayuserword[x*U:(x+1)*U,:]
		regiondayworduser = ssp.vstack(
			(regiondayworduser, rduw_x.T)
		)
	logger.debug("... cleaning up Filling (R x D x W, U) matrix")
	regiondayworduser = ssp.csr_matrix(regiondayworduser)
	return regiondayuserword,regiondayworduser
	