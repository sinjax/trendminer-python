import bivariate.crossvalidation.timeseriescv as tscv
import bivariate.dataset.billdata as billdata
import os

home = os.environ['HOME']
data_home = "%s/Dropbox/Experiments/twitter_uk_users_MATLAB"%home
user_mat_file = "user_vsr_for_polls_day_%d_%d_t.mat"
word_mat_file = "user_vsr_for_polls_day_%d_%d.mat"

task_file = "%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartII.mat"%home

start = 81
ndays = 20
end = start + ndays

user_file = os.sep.join([data_home,user_mat_file%(start,end)])
word_file = os.sep.join([data_home,word_mat_file%(start,end)])

folds = tscv.tsfi(ndays,ntest=2)
tasks = billdata.taskvals(task_file).mat()
user_col, word_col = billdata.suserdayword(user_file,word_file,ndays).mat()

for fold in folds:
	yparts = fold.parts(tasks)
	# At this stage the user_col and word_col matrices should be used to
	# optimise the lambda parameter for user and word learning respectively

	# This is easy for w, simply prepare the user_col matrix as per bill's 
	# formWordsMatrix function and go fourth
	# the next part is trickier, a single round of bilinear learning with
	# this lambda_w must proceed and that value of w1 must be used
	# to prepare the word_col matrix and find an optimal lambda_u

	# to achieve this it is important that a single step of the bilinear 
	# learner be exposed. Not a problem in itself.
