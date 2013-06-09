import bivariate.crossvalidation.timeseriescv as tscv
import bivariate.dataset.billdata as billdata
from bivariate.learner.spamsfunc import *
from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from pylab import *
import os
from IPython import embed
import logging;logger = logging.getLogger("root")


home = os.environ['HOME']
data_home = "%s/Dropbox/Experiments/twitter_uk_users_MATLAB"%home
user_mat_file = "user_vsr_for_polls_day_%d_%d_t.mat"
word_mat_file = "user_vsr_for_polls_day_%d_%d.mat"

task_file = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartII.mat"%home

start = 81
ndays = 20
end = start + ndays
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

user_file = os.sep.join([data_home,user_mat_file%(start,end)])
word_file = os.sep.join([data_home,word_mat_file%(start,end)])

folds = tscv.tsfi(ndays,ntest=2)
tasks = billdata.taskvals(task_file).mat(days=(start,end))
user_col, word_col = billdata.suserdayword(
	user_file,word_file,ndays
).mat(word_subsample=0.01)


learner = BatchBivariateLearner(w_spams,u_spams)

fold_i = 0
for fold in folds:
	logger.debug("Working on fold: %d"%fold_i)
	logger.debug("... preparing fold parts")
	Xparts,Yparts = BatchBivariateLearner.XYparts(fold,user_col,tasks)
	logger.debug("... optimising fold lambda")
	learner.optimise_lambda(w_lambdas,u_lambdas,Yparts,Xparts)
	logger.debug("... training fold")
	learner.process(Yparts.train_all,Xparts.train_all)
	fold_i += 1
