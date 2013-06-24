import bivariate.crossvalidation.timeseriescv as tscv
import bivariate.dataset.billdata as billdata
from bivariate.learner.spamsfunc import *
import bivariate.experiment.expstate as es
from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from pylab import *
import scipy.io as sio
import scipy.sparse as ssp
import os
from IPython import embed
import logging;logger = logging.getLogger("root")
import copy

user_mat_file = "user_vsr_for_polls_t.mat"
user_mat_corrected_file = "user_vsr_for_polls_t_corrected_%d->%d.mat"
word_mat_file = "user_vsr_for_polls.mat"

def prep_input_files(xargs):
	dh = xargs["data_home"]
	tdh = xargs["task_data_home"]
	if xargs["task_file"] is None: xargs["task_file"] = "%s/UK_data_for_experiment_PartII.mat"%tdh
	if xargs["tree_file"] is None: xargs["tree_file"] = "%s/UK_data_for_experiment_PartI.mat"%tdh
	if xargs["voc_file"] is None: xargs["voc_file"] = "%s/voc_matching_v2.mat"%tdh
	if xargs["user_file"] is None: xargs["user_file"] = os.sep.join([dh,user_mat_file])
	if xargs["user_file_corrected"] is None: xargs["user_file_corrected"] = os.sep.join([dh,user_mat_corrected_file%(xargs["start"],xargs["start"]+xargs["ndays"])])
	if xargs["word_file"] is None: xargs["word_file"] = os.sep.join([dh,word_mat_file])


	
def experiment(o):			
	logger.info("Reading initial data")
	start = o["start"];ndays = o["ndays"];end = start + ndays
	folds = tscv.tsfi(ndays,ntest=o['f_ntest'],nvalidation=o['f_nval'],ntraining=o['f_ntrain'])
	
	tasks = billdata.taskvals(o["task_file"])
	ndays_total = tasks.yvalues.shape[0]
	if o["user_file_corrected"] is None or not os.path.exists(o["user_file_corrected"]):
		logger.info("...Loading and correcting from source")
		if "voc_file" in o and not o["word_subsample"] < 1:
			logger.info("...Reading vocabulary")
			voc = billdata.voc(o["voc_file"]).voc()
			# voc = None
		else:
			voc = None
		logger.info("...Reading user days")
		user_col, word_col = billdata.suserdayword(
			o["user_file"],ndays_total,nwords=billdata.count_cols_h5(o["word_file"])
		).mat(days=(start,end),voc=voc)
		if o["user_file_corrected"] is not None:
			logger.info("...Saving corrected user_mat")
			sio.savemat(o["user_file_corrected"],{"data":user_col.data,"indices":user_col.indices,"indptr":user_col.indptr,"shape":user_col.shape})
	else:
		logger.info("...Loading corrected user_mat")
		# csc_matrix((data, indices, indptr), [shape=(M, N)])
		user_col_d = sio.loadmat(o["user_file_corrected"])
		user_col = ssp.csc_matrix((user_col_d["data"][:,0],user_col_d["indices"][:,0],user_col_d["indptr"][:,0]),shape=user_col_d["shape"])

	logger.info("...Reading task data")
	tasks = tasks.mat(days=(start,end))
	logger.info("...Reading tree file")
	tree = billdata.tree(o["tree_file"]).spamsobj()

	if o["word_subsample"] < 1 or o["user_subsample"] < 1:
		user_col=billdata.subsample(user_col,word_subsample=o["word_subsample"],user_subsample=o["user_subsample"],ndays=ndays)
	# At this point we've just loaded all the data
	# Prepare the optimisation functions
	u_lambdas = [float(x) for x in o['u_lambdas_str'].split(",")]
	w_lambdas = [float(x) for x in o['w_lambdas_str'].split(",")]
	u_lambdas = np.arange(*u_lambdas)
	w_lambdas = np.arange(*w_lambdas)
	spams_avail = {
		"tree":FistaTree(tree,**{
			"intercept": True,
			"loss":"square",
			"regul":"tree-l2",
			"it0":3,
			"max_it":150
		}),
		"flat":FistaFlat(**{
			"intercept": True,
			"loss":"square",
			"regul":"elastic-net",
			"it0":3,
			"max_it":150
		})
	}

	w_spams = copy.deepcopy(spams_avail[o["w_spams"]])
	u_spams = copy.deepcopy(spams_avail[o["u_spams"]])
	lambda_set = False
	if o["lambda_file"] is not None and os.path.exists(o["lambda_file"]):
		logger.info("... loading existing lambda")
		lambda_d = sio.loadmat(o["lambda_file"])
		w_spams.params["lambda1"] = lambda_d["w_lambda"][0][0]
		u_spams.params["lambda1"] = lambda_d["u_lambda"][0][0]
		lambda_set = True

	# Prepare the learner
	learner = BatchBivariateLearner(w_spams,u_spams)
	fold_i = 0
	es.exp(os.sep.join([o['exp_out'],"ds:politics_word:l1_user:l1_task:multi"]),fake=False)
	# Go through the folds!
	for fold in folds:
		es.state("fold_%d"%fold_i)
		logger.info("Working on fold: %d"%fold_i)
		logger.info("... preparing fold parts")
		Xparts,Yparts = BatchBivariateLearner.XYparts(fold,user_col,tasks)
		if not o["optimise_lambda_once"] or (o["optimise_lambda_once"] and not lambda_set):
			logger.debug("... Setting max it to optimisation mode: %d"%o["opt_maxit"])
			w_spams.params["max_it"] = o["opt_maxit"]
			u_spams.params["max_it"] = o["opt_maxit"]
			logger.info("... optimising fold lambda")
			ulambda,wlambda = learner.optimise_lambda(w_lambdas,u_lambdas,Yparts,Xparts)
			lambda_set = True
			if o["lambda_file"] is not None:
				logger.info("... saving optimised lambdas")
				sio.savemat(o["lambda_file"],{"w_lambda":wlambda[1],"u_lambda":ulambda[1]})
		logger.info("... training fold")
		logger.debug("... Setting max it to training mode: %d"%o["train_maxit"])
		w_spams.params["max_it"] = o["train_maxit"]
		u_spams.params["max_it"] = o["train_maxit"]
		learner.process(Yparts.train_all,Xparts.train_all,tests={"test":(Xparts.test,Yparts.test),"val_it":(Xparts.val_it,Yparts.val_it)})
		es.add(locals(),"fold_i","w_lambdas","u_lambdas","fold","Yparts","o")
		es.state()["w_spams_params"] = w_spams.params 
		es.state()["u_spams_params"] = u_spams.params
		logger.info("... Saving output")
		es.flush()
		fold_i += 1
		if o["f_maxiter"] is not None and fold_i >= o["f_maxiter"]: break

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	home = os.environ['HOME']
	data_home = "%s/Dropbox/Experiments/twitter_uk_users_MATLAB/"%home
	task_data_home = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/Data/"%home
	parser.add_option("--data-home", "--dh", dest="data_home",default=data_home,
					  help="root location where user_vsr_for_polls and user_vsr_for_polls_t can be found")
	parser.add_option("--task-data-home", "--tdh", dest="task_data_home", default=task_data_home,
					  help="root location where UK_data_for_experiment_PartII.mat etc. can be found")
	parser.add_option("-x", "--user-mat-file", dest="user_file",
					  help="File containing a sparse matrix (hdf5 or mat) of user/days in the columns and words in the rows")
	parser.add_option("--xc", "--user-mat-corrected-file", dest="user_file_corrected",default=None,
					  help="File containing a sparse matrix (hdf5 or mat) of user/days in the columns and words in the rows (corrected for vocabulary)")
	parser.add_option("--word-mat-file", dest="word_file",
					  help="File containing a sparse matrix (hdf5 or mat) of user/days in the columns and words in the rows")
	parser.add_option("-y", "--task", dest="task_file",
					  help="File containing the tasks in the cols and rows as days")
	parser.add_option("-t", "--tree", dest="tree_file",
					  help="File containing the tree for regularisation")
	parser.add_option("-v", "--voc", dest="voc_file",
					  help="File containing the vocabulary correction")
	parser.add_option("-s", "--start", dest="start",
					  help="The start day", default=81, type="int")
	parser.add_option("-d", "--n-days", dest="ndays",
					  help="The number of days", default=271, type="int")
	parser.add_option("--fold-ntest","--ft", dest="f_ntest",
					  help="The number of days in the test of each fold", default=5, type="int")
	parser.add_option("--fold-maxiter","--fm", dest="f_maxiter",
					  help="The max number of iterations (regardless of fold params)", type="int")
	parser.add_option("--fold-ntrain","--ftr", dest="f_ntrain",
					  help="The first number of training examples in first fold", default=190, type="int")
	parser.add_option("--fold-nval", "--fv",dest="f_nval",
					  help="The number of validation examples taken from each fold's training", default=60, type="int")
	parser.add_option("--u-lambdas", dest="u_lambdas_str",
					  help="The lambdas to search for u, comma seperater: start,end,gap", default="0.1,1,0.1")
	parser.add_option("--w-lambdas", dest="w_lambdas_str",
					  help="The lambdas to search for w, comma seperater: start,end,gap", default="1.0,3,0.1")
	parser.add_option("-o","--experiment-out", dest="exp_out",
					  help="Output location for the experiment", default="%s/Experiments/EMNLP2013"%home)
	parser.add_option("--ssw","--sub-sample-word", dest="word_subsample",
					  help="Choose some proportion of the words", type="float", default=1.)
	parser.add_option("--ssu","--sub-sample-user", dest="user_subsample",
					  help="Choose some proportion of the users", type="float", default=1.)
	parser.add_option("--wspm","--w-spams-mode", dest="w_spams",
					  help="How W should be optimised", default="flat", choices=["tree","flat"])
	parser.add_option("--uspm","--u-spams-mode", dest="u_spams",
					  help="How U should be optimised", default="flat", choices=["tree","flat"])
	parser.add_option("--lambda-file", "--lf", dest="lambda_file",
					  help="A file containing the lambda value of u and w. Setting this file forces this lambda to be used for all folds. Setting this file if it doesn't exist forces one round of optimisation.", default=None)
	parser.add_option("--optimise-lambda-once", "--olo", action="store_true", dest="optimise_lambda_once",
					  help="If set lambda is optimised once on the first fold and never again", default=False)
	parser.add_option("--optimise-maxit", "--omi", dest="opt_maxit", default=100,
					  help="The max iterations for optimisation of lambda",type="int")
	parser.add_option("--train-maxit", "--tmi", dest="train_maxit", default=1000,
					  help="The max iterations for training",type="int")


	(options, args) = parser.parse_args()
	options = vars(options)
	prep_input_files(options)
	experiment(options)