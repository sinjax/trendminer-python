import tabulate
from pylab import *
import argparse
import logging;logger = logging.getLogger("root")
import os
import sys
from IPython.frontend.terminal.embed import *
from bivariate.tools.utils import *
import cPickle as pickle
from scipy import io as sio
from ..learner.batch.regionuserwordlearne

def rmse_table(options):
	if not os.path.exists(args.exp): logger.error("Could not find experiment at location: %s"%args.exp)
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))

	folds = [x for x in os.listdir(args.exp) if "fold" in x]
	folds.sort()
	if args.force_fold >= 0: folds = [x for x in folds if "%d"%args.force_fold in x]
	folds = [os.sep.join([args.exp,x]) for x in folds]

	epoch_folds_dict = {}

	for fold_dir in folds:
		epochs = [x for x in os.listdir(fold_dir) if "epoch" in x]
		epochs.sort()
		if args.force_epoch >= 0: epochs = [x for x in epochs if "%d"%args.force_epoch in x]
		for epoch_name in epochs: