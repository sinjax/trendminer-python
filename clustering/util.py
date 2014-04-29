import re
from IPython import embed
from pylab import *

def textscan(fp,regexp,types=None):
	f = file(fp,"r")
	mats = None

	for line in f.readlines():
		line = line.strip()
		matches = re.match(regexp,line)
		groups = matches.groups()

		if mats == None:
			mats = [[transform(groups[x],types,x)] for x in range(len(groups))]
		else:
			for x in range(len(mats)):
				mats[x] += [transform(groups[x],types,x)]
	mats = [array(x) for x in mats]
	return mats

def transform(s,types,ind):
	if types == None: return float(s)
	else: return types[ind](s)
	