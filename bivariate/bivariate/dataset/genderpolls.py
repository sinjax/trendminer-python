import scipy.io as sio
from pylab import array
from IPython import embed

def transform(pollfile):
	"""
		pollfile - file containing polls

		return: 
			a tensor of polls of the shape Days x Tasks x Genders
	"""
	polls = sio.loadmat(pollfile)
	regkeys = [a for a,b in polls.items() if "male" in a or "female" in a]
	regkeys.sort()
	
	Yarr = [polls[x] for x in regkeys]
	Yarr = array([polls[x] for x in regkeys])
	
	Yarr = Yarr.transpose([1,2,0])

	return Yarr

