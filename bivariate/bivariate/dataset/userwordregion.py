import logging;logger = logging.getLogger("root")
from scipy import sparse as ssp
from IPython import embed
from pylab import array,hstack,ones
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
	dayuserwords_l = ssp.lil_matrix(dayuserwords)

	
	logger.debug("Filling (R x D x U, W) matrix")
	rows = []
	data = []
	for r in range(R):
		rusers = set(regionusermap[r])
		for n in range(N):
			for u in range(U):
				if u not in rusers: 
					rows += [[]]
					data += [[]]
				else:
					i = n * U + u
					rows += [dayuserwords_l.rows[i]]
					data += [dayuserwords_l.data[i]]
	
	regiondayuserword = ssp.lil_matrix((1,1),dtype=dayuserwords_l.dtype)
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
	