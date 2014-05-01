import os 
import argparse
import matrix_create as mcreate
from matrix_create import DATE_QUERY
from IPython import embed
import datetime
import time
from scipy import sparse as ssp
from ..rdfutil import *

parser = argparse.ArgumentParser(description='Turn a file into a matrix')
def exists_and_correct(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
    	try:
    		extracted_date = datetime.datetime.strptime(os.path.basename(arg).split(".")[0].split("_")[1],"%Y-%m-%d")
    	except Exception, e:
    		parser.error("The file %s must match the format: <anything>_YYYY-MM-DD.<anything>")
    		return
    	return arg  #return an open file handle

def file_exists(parser,f):
	if not os.path.exists(f):
		parser.error("The file %s does not exist."%f)
	else:
		return f
parser.add_argument(
	"files", help="Files to read", nargs='+',
	type=lambda x: exists_and_correct(parser,x)
)
parser.add_argument(
	"-o", dest="output",help="File to write", metavar="FILE", required=True
)
parser.add_argument(
	"-t", dest="type",help="Type of sparse matrix", default="csr"
)
def save_matlab(matrix,options):
	from scipy import io as sio
	sio.savemat(options.output_file,{options.name:matrix})
parser.add_argument(
	"-name", dest="name",help="The name in matlab", default="mat"
)
parser.set_defaults(save=save_matlab)
parser.add_argument(
	"-days", dest="dategraph",required=True,help="Day index to date",default=None,type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-words", dest="wordgraph",required=True,help="Word to index and class",default=None,type=lambda x: file_exists(parser,x)
)
parser.add_argument(
	"-users", dest="usergraph",required=True,help="User to index",default=None,type=lambda x: file_exists(parser,x)
)

parser.add_argument(
	"-daygroup", 
	dest="daygroup",
	help="How to group days",
	choices=["month"],
	default="month",
)

options = parser.parse_args()

sora_vs,sora_vsd = mcreate.create_sparse(options.files,**dict(options._get_kwargs()))

print "Saving all days..."
options.output_file = os.sep.join([options.output,"sora_vs"])
options.name = "sora_vs"
options.save(sora_vs,options)

options.output_file = os.sep.join([options.output,"sora_vsd"])
options.name = "sora_vsd"
options.save(sora_vsd,options)



if options.daygroup == "month":
	N,days_index = load_index(options.dategraph,DATE_QUERY)
	U = sora_vs.shape[0]/N
	index_days = dict([(y,x) for (x,y) in days_index.items()])
	def month_aggregate(ret):
		combined = {}
		for day in range(N):
			if not day in index_days: continue # sometimes there were no articles on weekends
			
			key = index_days[day][:7] # something like 2006-02
			rng = range(day*U,(day+1)*U)
			if key not in combined:
				combined[key] = ret[rng,:]
			else:
				combined[key] = combined[key] + ret[rng,:]
		combined_keys = combined.keys()
		combined_keys.sort()
		new_ret = ssp.vstack([combined[k] for k in combined_keys],format="csr")
		return new_ret
	print "Grouping days by month"
	month_sora_vs = month_aggregate(sora_vs)
	month_sora_vsd = month_aggregate(sora_vsd)
	
	options.output_file = "%s/month_sora_vs.mat"%(options.output)
	options.name = "month_sora_vs"
	options.save(month_sora_vs,options)

	options.output_file = "%s/month_sora_vsd.mat"%(options.output)
	options.name = "month_sora_vsd"
	options.save(month_sora_vsd,options)