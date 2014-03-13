from scipy import io as sio
from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")
from experimentfolds import *
from optparse import OptionParser
from ..learner.batch.regionuserwordlearner import SparseRUWLearner,prep_wspams,prep_uspams,prep_w_graphbit
import cPickle as pickle
import time
millis = int(round(time.time() * 1000))

parser = OptionParser()
parser.add_option("-x", "--user-word-mat", dest="xroot",
                  help="The root of the user word matrices", metavar="FILE")
parser.add_option("-y", "--polls", dest="polls",metavar="FILE",
                  help="The polls tensor file")
parser.add_option("-f", "--nfolds", dest="nfolds",default=1,
                  help="Number of folds")
parser.add_option("--n-training", dest="ntrain",default=48,
                  help="Number of training instances in the first fold")
parser.add_option("--n-validation", dest="nval",default=8,
                  help="Number of training instances in the first fold to be used as validation")
parser.add_option("--n-test", dest="ntest",default=5,
                  help="Number of test instances and the increment per fold")
parser.add_option("-d", "--ndays", dest="ndays", default=235,
                  help="Number of days")
parser.add_option("-w", "--w-spams", dest="w_spams_file", default=None, metavar="FILE",
                  help="Number of days")
parser.add_option("-n", "--nthreads", dest="nthreads", default=-1,
                  help="Number of spams threads")
parser.add_option("--u-lambda", dest="u_lambda", default=0.001,
                  help="The lambda given to the user regulariser")
parser.add_option("--w-lambda", dest="w_lambda", default=0.0001,
                  help="The lambda given to the word regulariser")
parser.add_option("--output", dest="output_root", default="out",
                  help="Time stamped output folder created here")
(options, args) = parser.parse_args()


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

experiments = prepare_folds(
	options.ndays,options.nfolds,
	step=int(options.ntest),t_size=int(options.ntrain),v_size=int(options.nval)
)
dataCache = {}
Y = sio.loadmat(options.polls)['regionpolls']
uw,wu = read_split_userwordregion(options.xroot,cache=dataCache,*[0])
W = uw.shape[1]
U = wu.shape[1]
R = wu.shape[0]/W
T = Y.shape[1]

w_spams_graphbit = None

if options.w_spams_file and os.path.exists(options.w_spams_file):
	w_spams_graphbit = pickle.load(file(options.w_spams_file,"rb"))
else:
	w_spams_graphbit = prep_w_graphbit(U,W,T,R)
	if options.w_spams_file:
		pickle.dump(w_spams_graphbit,file(options.w_spams_file,"wb"))

w_spams = prep_wspams(U,W,T,R,graphbit=w_spams_graphbit,lambda1=float(options.w_lambda))
u_spams = prep_uspams(lambda1=float(options.u_lambda))

w_spams.params['numThreads'] = int(options.nthreads)
u_spams.params['numThreads'] = int(options.nthreads)

fold_n = 0
for fold in experiments:
	fold_dir = os.sep.join([output_dir,"fold_%d"%fold_n])
	os.makedirs(fold_dir)
	def save_epoch(epoch):
		epoch_file = os.sep.join([fold_dir,"epoch_%d"%epoch["epoch"]])
		logger.debug("Saving Epoch: %s"%epoch_file)
		dump_compressed(epoch,epoch_file)
		
	learner = SparseRUWLearner(u_spams,w_spams, epoch_callback=save_epoch)
	training = fold['training']
	test = fold['test']
	validation = fold['validation']
	logger.debug("Reading training data...")
	Xtrain_uw,Xtrain_wu = read_split_userwordregion(options.xroot,cache=dataCache,*training)
	Ytrain = Y[training,:,:]
	logger.debug("Learning...")
	learner.learn(Xtrain_uw,Xtrain_wu,Ytrain)
	fold_n += 1
