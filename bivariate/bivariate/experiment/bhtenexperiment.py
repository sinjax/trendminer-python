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

for fold in folds:
	y_train = tasks[fold.train(),:];
	y_train_all = tasks[fold.train_all(),:];
	y_test = tasks[fold.test(),:];
	val = fold.val()
	split = len(val)/3
	y_val_param = tasks[val[:split*2],:];
	y_val_it = tasks[val[split*2:],:];
