from pylab import *
import wordcloud
import argparse
import logging;logger = logging.getLogger("root")
import os
import sys
from IPython.frontend.terminal.embed import *
from bivariate.tools.utils import *
import cPickle as pickle
from scipy import io as sio
from ..learner.batch.regionuserwordlearner import print_epoch_words,scored_epoch_words

def nfold_vis(args):
	if not os.path.exists(args.exp): logger.error("Could not find experiment at location: %s"%args.exp)

	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	

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
	rc('axes', color_cycle=['r', 'g', 'b'])
	for epoch in epochs:
		figdir = os.sep.join([out_dir,"%s"%epoch])
		if not os.path.exists(figdir): os.makedirs(figdir)
		for r in epoch_folds_dict[epoch]:
			figure()
			title("region: %s"%(region_map[r]))
			plot(epoch_folds_dict[epoch][r]['mean'].T,".",)
			plot(epoch_folds_dict[epoch][r]['correct'].T,"--")
			lines = plot(epoch_folds_dict[epoch][r]['estimate'].T,"-")
			legend(lines, [task_map[x] for x in range(len(lines))])
			figpath = os.sep.join([figdir,"region_%d.pdf"%r])

			savefig(figpath)

	show()
			

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
def stats(args):
	if not os.path.exists(args.exp):
		logger.error("Could not find experiment at location: %s"%args.exp)

	folds = [x for x in os.listdir(args.exp) if "fold" in x]
	folds.sort()
	if args.force_fold >= 0: folds = [x for x in folds if "%d"%args.force_fold in x]
	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	if not os.path.exists(out_dir): os.makedirs(out_dir)

	epoch_folds_dict = {}
	logger.info("Number of folds: %d"%len(folds))
	for fold in folds:
		fold_dir = os.sep.join([args.exp,fold])
		epochs = [x for x in os.listdir(fold_dir) if "epoch" in x]
		epochs.sort()
		if args.force_epoch >= 0: epochs = [x for x in epochs if "%d"%args.force_epoch in x]
		for epoch_name in epochs:
			logger.info("Fold: %s, Epoch: %s"%(fold,epoch_name))
			epoch_path = os.sep.join([fold_dir,epoch_name])
			epoch = load_compressed(epoch_path)
			u_hat = epoch['u_hat']
			w_hat = epoch['w_hat']
			selected_users = sum(abs(u_hat).sum(axis=(0,1)) > 0)
			selected_words = sum(abs(w_hat).sum(axis=(0,1)) > 0)
			logger.info("Total Selected Users: %d"%selected_users)
			logger.info("Total Selected Words: %d"%selected_words)
			R = u_hat.shape[0]
			for r in range(R):
				logger.info("Region %d (%s)"%(r,region_map[r]))
				logger.info("Selected Users: %d"%sum(abs(u_hat[r,:,:]).sum(axis=0) > 0))
				logger.info("Selected Words: %d"%sum(abs(w_hat[r,:,:]).sum(axis=0) > 0))

def important_words(args):
	if not os.path.exists(args.exp):
		logger.error("Could not find experiment at location: %s"%args.exp)
	
	voc = sio.loadmat(args.vocabulary)[args.voc_key]
	
	folds = [x for x in os.listdir(args.exp) if "fold" in x]
	folds.sort()
	if args.force_fold >= 0: folds = [x for x in folds if "%d"%args.force_fold in x]
	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	if not os.path.exists(out_dir): os.makedirs(out_dir)

	
	logger.info("Number of folds: %d"%len(folds))
	for fold in folds:
		fold_dir = os.sep.join([args.exp,fold])
		epochs = [x for x in os.listdir(fold_dir) if "epoch" in x]
		epochs.sort()
		if args.force_epoch >= 0: epochs = [x for x in epochs if "%d"%args.force_epoch in x]
		for epoch_name in epochs:
			logger.info("Fold: %s, Epoch: %s"%(fold,epoch_name))
			epoch_path = os.sep.join([fold_dir,epoch_name])
			epoch = load_compressed(epoch_path)
			w_hat = epoch['w_hat']
			selected_words = sum(abs(w_hat).sum(axis=(0,1)) > 0)
			logger.info("Total Selected Words: %d"%selected_words)
			print_epoch_words(epoch,voc,args.nwords,args.force_region)

def words_cloud(args):

	voc = sio.loadmat(args.vocabulary)[args.voc_key]
	
	folds = [x for x in os.listdir(args.exp) if "fold" in x]
	folds.sort()

	if args.force_fold >= 0: folds = [x for x in folds if "%d"%args.force_fold in x]
	exp_opts = pickle.load(file(os.sep.join([args.exp,"cmd_opts"]),"rb"))
	out_dir = os.sep.join([args.out,"u_lambda=%s"%exp_opts.u_lambda,"w_lambda=%s"%exp_opts.w_lambda])
	if not os.path.exists(out_dir): os.makedirs(out_dir)

	logger.info("Number of folds: %d"%len(folds))
	for fold in folds:
		fold_dir = os.sep.join([args.exp,fold])
		epochs = [x for x in os.listdir(fold_dir) if "epoch" in x]
		epochs.sort()
		if args.force_epoch >= 0: epochs = [x for x in epochs if "%d"%args.force_epoch in x]
		for epoch_name in epochs:
			ef_out_dir = os.sep.join([out_dir,fold,epoch_name])
			if not os.path.exists(ef_out_dir): os.makedirs(ef_out_dir)
			epoch_path = os.sep.join([fold_dir,epoch_name])
			epoch = load_compressed(epoch_path)

			region_task_words = scored_epoch_words(epoch,voc,args.nwords,args.force_region)
			
			for r in region_task_words:
				for t in region_task_words[r]:
					for pn in region_task_words[r][t]:
						if args.for_wordle:
							outname = "%s_r%dt%d.wordle"%(pn,r,t)
							outfile = os.sep.join([ef_out_dir,outname])
							f = file(outfile,"w")
							for word,score in region_task_words[r][t][pn]:
								f.write("%s: %2.5f\n"%(word,score))
							f.close()
						else:
							outname = "%s_r%dt%d.png"%(pn,r,t)
							fit_rtw = wordcloud.fit_words(region_task_words[r][t][pn],font_path=args.font)
							wordcloud.draw(fit_rtw, os.sep.join([ef_out_dir,outname]),font_path=args.font)

def main():
	parser = argparse.ArgumentParser(description='Visualise a region experiment')
	parser.add_argument("-exp", dest="exp",default="./", help="Location of the experiment, must contain the log, config and fold dirs")
	parser.add_argument("-out", dest="out",default="./out", help="Location of the experiment, must contain the log, config and fold dirs")
	parser.add_argument("-fold", dest="force_fold",default=-1,type=int,help="Force the fold.")
	parser.add_argument("-epoch", dest="force_epoch",default=-1,type=int,help="Force the epoch to extract graphs from. Defaults to all epochs")
	parser.add_argument("-region", dest="force_region",default=-1,type=int,help="Force the region to extract data from")
	
	subparsers = parser.add_subparsers()
	parser_windowed = subparsers.add_parser('nfold')
	parser_windowed.set_defaults(vis=nfold_vis)
	
	parser_windowed = subparsers.add_parser('worduserstats')
	parser_windowed.set_defaults(vis=stats)
	
	parser_windowed = subparsers.add_parser('words')
	parser_windowed.set_defaults(vis=important_words)
	parser_windowed.add_argument("-voc", dest="vocabulary",default=None,type=str,required=True,help="Location of the vocabulary file to load words from")
	parser_windowed.add_argument("-vockey", dest="voc_key",default="voc_filtered",type=str,help="Key to get vocabulary from")
	parser_windowed.add_argument("-n", dest="nwords",default=20,type=int,help="Number of words to display per region")

	parser_windowed = subparsers.add_parser('wordcloud')
	parser_windowed.set_defaults(vis=words_cloud)
	parser_windowed.add_argument("-voc", dest="vocabulary",default=None,type=str,required=True,help="Location of the vocabulary file to load words from")
	parser_windowed.add_argument("-vockey", dest="voc_key",default="voc_filtered",type=str,help="Key to get vocabulary from")
	parser_windowed.add_argument("-n", dest="nwords",default=20,type=int,help="Number of words to display per region")
	parser_windowed.add_argument("-font", dest="font",default="/usr/local/texlive/2012/texmf-dist/fonts/truetype/public/droid/DroidSansMono.ttf",type=str,help="Font to draw")
	parser_windowed.add_argument("-wordle", dest="for_wordle",default=False,action="store_true")
	
	

	options = parser.parse_args()

	options.vis(options)


if __name__ == '__main__':
	main()
