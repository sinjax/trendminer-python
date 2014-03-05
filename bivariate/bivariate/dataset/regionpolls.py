import scipy.io as sio
from pylab import array

def transform(regionpollfile):
	"""
		regionpollfile - file containing polls

		return: 
			a tensor of polls of the shape Days x Tasks x Regions
	"""
	polls = sio.loadmat(regionpollfile)
	regkeys = [a for a,b in polls.items() if "region" in a]
	regkeys.sort()
	
	Yarr = [polls[x] for x in regkeys]
	Yarr = array([polls[x] for x in regkeys])
	
	Yarr = Yarr.transpose([1,2,0])

	return Yarr
