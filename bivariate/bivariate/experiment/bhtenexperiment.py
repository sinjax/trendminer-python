import bivariate.crossvalidation.timeseriescv as tscv
import bivariate.dataset.billdata as billdata
from bivariate.learner.spamsfunc import *
import bivariate.experiment.expstate as es
from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from pylab import *
import os
from IPython import embed
import logging;logger = logging.getLogger("root")

logger.info("Reading initial data")
home = os.environ['HOME']
data_home = "%s/Dropbox/Experiments/twitter_uk_users_MATLAB"%home
user_mat_file = "user_vsr_for_polls_day_%d_%d_t.mat"
word_mat_file = "user_vsr_for_polls_day_%d_%d.mat"

task_file = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartII.mat"%home
tree_file = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartI.mat"%home
voc_file = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/voc_matching_v2.mat"%home

start = 81;ndays = 20;end = start + ndays

user_file = os.sep.join([data_home,user_mat_file%(start,end)])
word_file = os.sep.join([data_home,word_mat_file%(start,end)])

folds = tscv.tsfi(ndays,ntest=2)
logger.info("Reading task data")
tasks = billdata.taskvals(task_file).mat(days=(start,end))
# tree = billdata.tree(tree_file).spamsobj()
logger.info("Reading vocabulary")
voc = billdata.voc(voc_file).voc()
user_col, word_col = billdata.suserdayword(
	user_file,word_file,ndays
).mat(voc=voc)

user_col,word_col=billdata.subsample(user_col,word_subsample=0.001,user_subsample=0.001,ndays=ndays)
# At this point we've just loaded all the data
# Prepare the optimisation functions
u_lambdas = np.arange(0.1,1,0.1)
w_lambdas = np.arange(0.1,2,0.1)
w_spams = FistaFlat(**{
	"intercept": True,
	"loss":"square",
	"regul":"l1",
	"it0":10,
	"max_it":1000
})
u_spams = FistaFlat(**{
	"intercept": True,
	"loss":"square",
	"regul":"l1"
})

# Prepare the learner
learner = BatchBivariateLearner(w_spams,u_spams)
fold_i = 0
es.exp("%s/Experiments/EMNLP2013/ds:politics_word:l1_user:l1_task:multi"%home)
# Go through the folds!
for fold in folds:
	es.state("fold_%d"%fold_i)
	logger.info("Working on fold: %d"%fold_i)
	logger.info("... preparing fold parts")
	Xparts,Yparts = BatchBivariateLearner.XYparts(fold,user_col,tasks)
	logger.info("... optimising fold lambda")
	learner.optimise_lambda(w_lambdas,u_lambdas,Yparts,Xparts)
	logger.info("... training fold")
	learner.process(Yparts.train_all,Xparts.train_all)

	es.add(locals(),"fold_i","w_lambdas","u_lambdas","fold")
	es.state()["w_spams_params"] = w_spams.params 
	es.state()["u_spams_params"] = u_spams.params
	logger.info("Saving output")
	es.flush()
	fold_i += 1
	break
