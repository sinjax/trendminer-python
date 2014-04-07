import logging;logger = logging.getLogger("root")
from scipy import sparse as ssp
from IPython import embed
from pylab import array,hstack,ones,np
from bivariate.tools.utils import h5py_sparse_row, h5py_save_sparse_row
import shutil
import os
from scipy import io as sio
from userwordregion import *
GENDERS = ["female","male"]
def load_gender_map(gender_map_file):
	"""
	Load a file of the format:
		userid {0,1} 

	where 1 == male, 0 == female

	into a map of the format:

		map[userid] = {"male","female"}
	"""
	gmf = file(gender_map_file,"r")
	user_gender = {}
	for line in gmf.readlines():
		if len(line) == 0: continue
		user,gender_number = line.split(" ")
		user_gender[int(user)] = GENDERS[int(gender_number)]
	return user_gender

def read_split_userwordgender(inp,*days,**xargs):
	cache = None; vkeep = None
	if "cache" in xargs:  cache = xargs["cache"]
	if not os.path.exists(inp): raise Exception("Input does not exist")
	if "voc_keep" in xargs: vkeep = xargs['voc_keep']

	allmats = {}
	allmats["worduser"] = []
	allmats["userword"] = []
	for gender in os.listdir(inp):
		rdir = os.sep.join([inp,gender])
		for n in days:
			cache_name = "%sn%d"%(gender,n)
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

def region_to_gender(region_root,gender_root,gender_map):
	if os.path.exists(gender_root):
		shutil.rmtree(gender_root)
	os.makedirs(gender_root)
	N = len(os.listdir(os.sep.join([region_root,"r0"])))
	
	d0 = read_split_userwordregion_regionflat(region_root,0)

	U = d0.shape[1]
	W = d0.shape[0]

	for u in range(U):
		if u not in gender_map:
			gender_map[u] = "none"


	for day in range(N):
		logger.debug("Loading worduser for day: %d"%day)
		day_worduser = read_split_userwordregion_regionflat(region_root,day)
		dwul = day_worduser.T.tolil()
		mats = {
			"male":ssp.lil_matrix((U,W),dtype=float),
			"female":ssp.lil_matrix((U,W),dtype=float),
			"none":ssp.lil_matrix((U,W),dtype=float)
		}
		for x in mats:
			outpath = os.sep.join([gender_root,x])
			if not os.path.exists(outpath): os.makedirs(outpath)
		for u in range(U):
			if len(dwul.data[u]) == 0: continue
			mats[gender_map[u]].data[u] = dwul.data[u]
			mats[gender_map[u]].rows[u] = dwul.rows[u]

		for x,y in mats.items():
			outpath = os.sep.join([gender_root,x,"n%d.mat"%day])
			subX = {
				"userword": ssp.csc_matrix(y),
				"worduser": ssp.csc_matrix(y.T),
			}
			sio.savemat(outpath, subX)