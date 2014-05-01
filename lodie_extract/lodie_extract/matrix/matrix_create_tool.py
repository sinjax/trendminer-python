import os 
import argparse
import matrix_create as mcreate
from IPython import embed
import datetime
import time
from scipy import sparse as ssp
parser = argparse.ArgumentParser(description='Turn a file into a matrix')
def file_exists(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')  #return an open file handle
parser.add_argument(
	"-i", dest="input",help="File to read", metavar="FILE", required=True, 
	type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-o", dest="output",help="File to write", metavar="FILE", required=True
)
parser.add_argument(
	"-t", dest="type",help="Type of sparse matrix", default="csr"
)
def save_matlab(matrix,options):
	# from utils import h5py_save_sparse_row
	# h5py_save_sparse_row(options.output,{options.name:matrix},compression=None)
	from scipy import io as sio
	sio.savemat(options.output,{options.name:matrix})
parser.add_argument(
	"-name", dest="name",help="The name in matlab", default="mat"
)
parser.set_defaults(save=save_matlab)
parser.add_argument(
	"-daymap", dest="daymap",help="Day index to date",default=None,type=lambda x: file_exists(parser,x)
)

parser.add_argument(
	"-daygroup", 
	dest="daygroup",
	help="How to group days",
	choices=["month"],
	default="month",
)

options = parser.parse_args()

ret = mcreate.create_sparse(options.input,**dict(options._get_kwargs()))

print "Saving all days..."
options.save(ret,options)

if options.daymap:
	if options.daygroup == "month":
		print "Grouping days by month"
		lines = options.daymap.readlines()
		lines = [x.split(" ") for x in lines]
		index_dt = [ int(x) for (x,y) in lines ]
		N = max(index_dt) + 1
		U = ret.shape[0]/N
		first_day = datetime.datetime.strptime("2006-02-01","%Y-%m-%d").utctimetuple()
		day_in_millis = 24 * 60 * 60
		index_to_datetime = lambda index: (
			datetime.datetime.fromtimestamp(time.mktime(first_day) + day_in_millis*index)
		)

		combined = {}
		for day in range(N):
			a = index_to_datetime(day)
			key = int("%d%02d"%(a.year,a.month))
			rng = range(day*U,(day+1)*U)
			if key not in combined:
				combined[key] = ret[rng,:]
			else:
				combined[key] = combined[key] + ret[rng,:]
		combined_keys = combined.keys()
		combined_keys.sort()
		new_ret = ssp.vstack([combined[k] for k in combined_keys],format="csr")
		
		options.output = "%s/month_%s"%(os.path.abspath(os.path.join(options.output, os.pardir)),os.path.basename(options.output))
		options.name = "month_" + options.name
		options.save(new_ret,options)
		