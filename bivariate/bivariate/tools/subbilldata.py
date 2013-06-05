import bivariate.dataset.billdata as bd
import scipy.io
import os

import logging as logger
import logging.config

logger.config.fileConfig("logconfig.ini")

# Where the data is coming from and going
user_home = os.environ['HOME']
data_home = "%s/Dropbox/Experiments/twitter_uk_users_MATLAB"%user_home

# load the polls (for the number of days)
p2 = scipy.io.loadmat("%s/Dropbox/TrendMiner/Collaboration/EMNLP_2013/MATLAB_v2/UK_data_for_experiment_PartII.mat"%user_home);
yvalues = p2["polls_vi_cum"][p2["polls_index"]-1,:]

# Load the big dataset
udw = bd.suserdayword(
	"%s/user_vsr_for_polls_t.mat"%data_home,
	"%s/user_vsr_for_polls.mat"%data_home,
	yvalues.shape[0]
)

# extract these days and save
start = 81
delta = 20
end = start + delta
ua,wa = udw.mat(days=(start,end))
matname = "user_vsr_for_polls_day_%d_%d"%(start,end)
outuser = "%s/%s_t.mat"%(data_home,matname)
outword = "%s/%s.mat"%(data_home,matname)
logger.debug("Saving days %d to %d (User Col matrix)"%(start,end))
bd.savesparse(ua,outuser,"%s_t"%matname)
logger.debug("Saving days %d to %d (Word Col matrix)"%(start,end))
bd.savesparse(wa,outword,"%s"%matname)