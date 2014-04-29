from tabulate import tabulate
from pylab import *
import argparse
import logging;logger = logging.getLogger("root")
import os
import sys
from IPython.frontend.terminal.embed import *
from bivariate.tools.utils import *
import cPickle as pickle
from scipy import io as sio
from IPython import embed
from ...learner.batch.regionuserwordlearner import print_epoch_words,scored_epoch_words
# from ..regionukpollvis import region_map,task_map,short_region_map,short_task_map

region_map = {
	0 : "South England",
	1 : "London",
	2 : "Midlands",
	3 : "North England",
	4 : "Scotland",
}

task_map = {
	0 : "Conservative",
	1 : "Labour",
	2 : "Lib Dem"
}
short_region_map = {
	0 : "Se",
	1 : "L",
	2 : "M",
	3 : "N",
	4 : "Sc",
	-1: "$\\mu$"
}

gender_map = {
	0 : "F",
	1 : "M",
	-1: "$\\mu$"
}

short_task_map = {
	0 : "CON",
	1 : "LAB",
	2 : "LBD"
}
def sse(a,b):
	return pow(a - b,2)
def rmse_table(args):
	if not os.path.exists(args.exp): logger.error("Could not find experiment at location: %s"%args.exp)

	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	

	folds = [x for x in os.listdir(args.exp) if "fold" in x]
	folds.sort()
	if args.force_fold >= 0: folds = [x for x in folds if "%d"%args.force_fold in x]
	folds = [os.sep.join([args.exp,x]) for x in folds]

	epoch_folds_dict = {}
	epoch_name = None
	R = None
	N = None
	T = None
	for fold_dir in folds:
		epochs = [x for x in os.listdir(fold_dir) if "epoch" in x]
		epochs.sort()
		if args.force_epoch >= 0: epochs = [x for x in epochs if "%d"%args.force_epoch in x]
		
		epoch_name = epochs[0]

		epoch_path = os.sep.join([fold_dir,epoch_name])
		epoch_folds = epoch_folds_dict.get(epoch_name,{})
		epoch = load_compressed(epoch_path)
		Y_est = epoch['Y_estimate'][0] # something weird happened to make the array wrap wrong
		Y_corr = epoch['Y_correct']
		R = Y_est.shape[2] #number of regions
		N = Y_est.shape[0] #number of days
		T = Y_est.shape[1] #number of tasks
		
		for r in range(R):
			region_folds = epoch_folds.get(r,{})
			e = region_folds.get("estimate",zeros([T,0]))
			c = region_folds.get("correct",zeros([T,0]))
			m = region_folds.get("mean",zeros([T,0]))

			region_folds["estimate"] = hstack([e,Y_est[:,:,r].T])
			region_folds["correct"] = hstack([c,Y_corr[:,:,r].T])
			region_folds["mean"] = hstack([m] + [epoch['Y_mean'][:,r:r+1]] * N)
			epoch_folds[r] = region_folds
		epoch_folds_dict[epoch_name] = epoch_folds

	region_results = epoch_folds_dict[epoch_name]
	bggr_key = "BGGR"
	mean_key = "$\\mathbf{B_\\mu}$"
	last_key = "$\\mathbf{B_\\text{last}}$"
	key_order = [mean_key,last_key,bggr_key]
	task_regions = {}
	task_regions[bggr_key] = dict([(t,dict([(r,None) for r in range(R)])) for t in range(T)])
	task_regions[mean_key] = dict([(t,dict([(r,None) for r in range(R)])) for t in range(T)])
	task_regions[last_key] = dict([(t,dict([(r,None) for r in range(R)])) for t in range(T)])
	
	
	for r in region_results:
		err = sse(region_results[r]['correct'], region_results[r]['estimate'])
		mean_err = sse(region_results[r]['correct'], region_results[r]['mean'])
		rolled = roll(region_results[r]['correct'],1,axis=1)
		roll_err = sse(region_results[r]['correct'], rolled)
		for t in range(T):
			task_regions[bggr_key][t][r] = sqrt(mean(err[t,:]))
			task_regions[mean_key][t][r] = sqrt(mean(mean_err[t,:]))
			task_regions[last_key][t][r] = sqrt(mean(roll_err[t,:]))
	
	mode_map = short_region_map
	if len(region_results) == 2: mode_map = gender_map
	# add the region headers
	headers = [mode_map[r] for t in task_regions[bggr_key] for r in task_regions[bggr_key][t].keys() + [-1] ]
	headers += ["$\\mathbf{\\mu_{\\text{all}}}$"]

	table = []
	number_fmt = "%03.1f"
	for x in key_order:
		task_region = task_regions[x]
		to_add = [x]
		for t in task_region:
			for r in task_region[t]:
				to_add += [number_fmt%task_region[t][r]]
			to_add += [number_fmt%mean([x for x in task_region[t].values()])]
		to_add += [number_fmt%mean([x for t in task_region for x in task_region[t].values()])]
		table += [to_add]
	
	print tabulate(table,headers,tablefmt=args.table_format,floatfmt=".1f")