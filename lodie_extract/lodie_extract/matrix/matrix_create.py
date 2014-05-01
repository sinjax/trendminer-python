from pylab import *
from scipy import sparse as ssp
from scipy import io as sio
from IPython import embed
import sys

def create_sparse(f,**xargs):
	
	lines = f.readlines()
	# entries = zeros((len(lines),4),dtype=int)
	entries = []
	i = 0
	ndays = 0
	nusers = 0
	nwords = 0
	for line in lines:
		if len(line) == 0: continue
		if i % 10000 == 0:
			sys.stdout.write("Reading line: %d \r" % (i) )
			sys.stdout.flush()
		i += 1
		line = line.strip()
		parts,count = line.split("\t")
		parts_split = parts.split(" ")
		if len(parts_split) == 2:
			parts_split += [0]
		day,user,word = parts_split
		entries += [[int(day),int(user),int(word),int(count)]]
		ndays = max(entries[-1][0],ndays)
		nusers = max(entries[-1][1],nusers)
		nwords = max(entries[-1][2],nwords)
	
	nusers += 1
	ndays += 1
	nwords += 1
	M = (nusers) * (ndays)
	N = (nwords)
	print "Done reading, creating matrix: %d x %d "%(M,N)
	out = ssp.dok_matrix((M,N))
	mkey = (lambda d,u: (d * nusers) + u)
	for entry in entries:
		day,user,word,count = entry
		out[mkey(day,user),word] = count
	out = getattr(out,"to%s"%xargs['type'])()
	return out
	
