from scipy import io as sio
from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")
from experimentfolds import *
# from optparse import OptionParser
import argparse
from ..learner.batch.regionuserwordlearner import SparseRUWLearner,prep_wspams,prep_uspams,prep_w_graphbit,print_epoch_words
import cPickle as pickle
import time
import sys
from pylab import mean

millis = int(round(time.time() * 1000))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-x", "--user-word-mat", dest="xroot",
                  help="The root of the user word matrices", metavar="FILE")
parser.add_argument("-y", "--polls", dest="polls",metavar="FILE",
                  help="The polls tensor file")
parser.add_argument("-f", "--nfolds", dest="nfolds",default=1,type=int,
                  help="Number of folds")
parser.add_argument("-e", "--epochs", dest="nepochs",default=10,type=int,
                  help="Number of epochs of the bilinear model ")
prepare_fold_args(parser)
# parser.add_argument("--n-training", dest="ntrain",default=48,
#                   help="Number of training instances in the first fold")
# parser.add_argument("--n-validation", dest="nval",default=8,
#                   help="Number of training instances in the first fold to be used as validation")
# parser.add_argument("--n-test", dest="ntest",default=5,
#                   help="Number of test instances and the increment per fold")
parser.add_argument("-d", "--ndays", dest="ndays", default=235,
                  help="Number of days")
parser.add_argument("-w", "--w-spams", dest="w_spams_file", default=None, metavar="FILE",
                  help="Number of days")
parser.add_argument("-n", "--nthreads", dest="nthreads", default=-1,
                  help="Number of spams threads")
parser.add_argument("--u-lambda", dest="u_lambda", default=0.001,
                  help="The lambda given to the user regulariser")
parser.add_argument("--w-lambda", dest="w_lambda", default=0.0001,
                  help="The lambda given to the word regulariser")
parser.add_argument("--output", dest="output_root", default="out",
                  help="Time stamped output folder created here")
parser.add_argument("--voc-keep", dest="vocabulary_keep", default=None,
                  help="A vector of word indices to keep")
parser.add_argument("--voc-keep-key", dest="vocabulary_keep_key", default="keep_index",
                  help="If a keep index is provided, what key should be used to find it")
parser.add_argument("--voc", dest="vocabulary", default=None,
                  help="The words being learnt against")
parser.add_argument("--voc-key", dest="voc_key", default="voc_filtered",
                  help="If the voc index is provided, what key")
parser.add_argument("--word-group-limit", dest="word_group_limit", default=None,
                  help="Arbitrarily choose the first groups to this count")
options = parser.parse_args()


voc_keep = None
voc = None
if options.vocabulary is not None and options.vocabulary_keep is None: options.vocabulary_keep = options.vocabulary

if options.vocabulary_keep: voc_keep = sio.loadmat(options.vocabulary_keep)[options.vocabulary_keep_key][0,:] - 1
if options.vocabulary: voc = sio.loadmat(options.vocabulary)[options.voc_key]

# Prepare the output location and change the log file location

if not os.path.exists(options.output_root): os.makedirs(options.output_root)
output_dir = os.sep.join([options.output_root,str(millis)])
os.makedirs(output_dir)
fhandle = logger.root.handlers[0]
logpath = os.sep.join([output_dir,os.path.basename(fhandle.baseFilename)])
mod_fhandler = logging.FileHandler(logpath)
mod_fhandler.setFormatter(fhandle.formatter)
mod_fhandler.setLevel(fhandle.level)
logging.root.addHandler(mod_fhandler)
logger.debug("Experiment started, output directory: %s"%output_dir)

# save the options
pickle.dump(options,file(os.sep.join([output_dir,"cmd_opts"]),"w"))

# Load a single day of the experiment so we can create the graph regulariser parameters
dataCache = {}
Y = sio.loadmat(options.polls)['regionpolls']
uw,wu = read_split_userwordregion(options.xroot,cache=dataCache,voc_keep=voc_keep,*[0])
W = uw.shape[1]
if voc_keep is not None and W > len(voc_keep):
	logger.error("Vocabulary size does not match the voc_keep index size, exiting")
	sys.exit()
U = wu.shape[1]
R = wu.shape[0]/W
T = Y.shape[1]

w_spams_graphbit = None
# ... Try to load and cache the graph regulariser
while w_spams_graphbit is None:
	if options.w_spams_file and os.path.exists(options.w_spams_file):
		w_spams_graphbit = pickle.load(file(options.w_spams_file,"rb"))
		if W * T * R != w_spams_graphbit[0]['groups_var'].shape[0]:
			logger.error("Loaded word graph group does not match vocabulary size, reloading, deleting old")
			w_spams_graphbit = None
			os.remove(options.w_spams_file)
	else:
		w_spams_graphbit = prep_w_graphbit(W,T,R)
		if options.w_spams_file:
			logger.error("Saving w_spams to: %s"%options.w_spams_file)
			pickle.dump(w_spams_graphbit,file(options.w_spams_file,"wb"))

if options.word_group_limit is not None:
	wgl = int(options.word_group_limit)
	new_w_spams_graphbit = [
		{
			"groups": ssp.csc_matrix(w_spams_graphbit[0]["groups"][:wgl,:wgl],dtype=bool), 
			"eta_g": w_spams_graphbit[0]["eta_g"][:wgl], 
			"groups_var": ssp.csc_matrix(w_spams_graphbit[0]["groups_var"][:,:wgl],dtype=bool), 
		},
		w_spams_graphbit[1][:wgl]
	]
	w_spams_graphbit = new_w_spams_graphbit

w_spams = prep_wspams(W,T,R,graphbit=w_spams_graphbit,lambda1=float(options.w_lambda))
# w_spams = prep_uspams(lambda1=float(options.u_lambda)) # Useful for test, much faster
u_spams = prep_uspams(lambda1=float(options.u_lambda))


# Prepare the experiments, and let's go!
experiments = options.folds(options)

fold_n = 0
for fold in experiments:
	fold_dir = os.sep.join([output_dir,"fold_%d"%fold_n])
	os.makedirs(fold_dir)
	def save_epoch(epoch):
		epoch_file = os.sep.join([fold_dir,"epoch_%d"%epoch["epoch"]])
		logger.debug("Saving Epoch: %s"%epoch_file)
		if voc is not None:
			print_epoch_words(epoch,voc)
		epoch['Y_estimate'] = learner.predict(Xtest_wu,epoch['u_hat'],epoch['w_hat'],epoch['b_hat'],len(test))
		epoch['Y_correct'] = Ytest
		epoch['Y_se'] = pow(epoch['Y_correct'] - epoch['Y_estimate'],2)
		epoch['Y_mse'] = mean(epoch['Y_se'])
		epoch['fold'] = fold
		logger.debug("Epoch_%d, Fold%d, mse: %2.2f"%(epoch["epoch"],fold_n,epoch['Y_mse']))
		dump_compressed(epoch,epoch_file)
		
	learner = SparseRUWLearner(u_spams,w_spams, epoch_callback=save_epoch, epochs=options.nepochs)
	training = fold['training']
	test = fold['test']
	validation = fold['validation']
	logger.debug("Reading training data...")
	Xtrain_uw,Xtrain_wu = read_split_userwordregion(options.xroot,voc_keep=voc_keep,cache=dataCache,*training)
	logger.debug("Reading test data...")
	Xtest_uw,Xtest_wu = read_split_userwordregion(options.xroot,voc_keep=voc_keep,cache=dataCache,*test)
	Ytrain = Y[training,:,:]
	Ytest = Y[test,:,:]
	logger.debug("Learning...")
	learner.learn(Xtrain_uw,Xtrain_wu,Ytrain)
	fold_n += 1
