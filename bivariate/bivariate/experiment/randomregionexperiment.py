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
from pylab import mean,np,random,norm
np.random.seed(1)
millis = int(round(time.time() * 1000))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-R",default=3 ,type=int,help="Number of regions")
parser.add_argument("-T",default=5 ,type=int,help="Number of tasks")
parser.add_argument("-U",default=7 ,type=int,help="Number of users")
parser.add_argument("-W",default=11,type=int,help="Number of words")
parser.add_argument("-N", "--ndays", dest="ndays", default=13,help="Number of days",type=int)
parser.add_argument("-f", "--nfolds", dest="nfolds",default=1,type=int,help="Number of folds")
parser.add_argument("-e", "--epochs", dest="nepochs",default=10,type=int,help="Number of epochs of the bilinear model ")
prepare_fold_args(parser)
parser.add_argument("--u-lambda", dest="u_lambda", default=0.001,help="The lambda given to the user regulariser")
parser.add_argument("--w-lambda", dest="w_lambda", default=0.0001,help="The lambda given to the word regulariser")
parser.add_argument("--output", dest="output_root", default="out",help="Time stamped output folder created here")
parser.add_argument("-fc", dest="force_mean_center", default=False, action='store_true',help="Remove the mean from the response variable")
options = parser.parse_args()

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
pickle.dump(options,file(os.sep.join([output_dir,"cmd_opts"]),"w"))

U,W,N,T,R = options.U,options.W,options.ndays,options.T,options.R

w_spams_graphbit = prep_w_graphbit(W,T,R)
w_spams = prep_wspams(W,T,R,graphbit=w_spams_graphbit,lambda1=float(options.w_lambda))
u_spams = prep_uspams(lambda1=float(options.u_lambda))

# the weights we aim to learn
u = np.random.random((R, T, U))
w = np.random.random((R, T, W))
b = np.random.random((T, R))

# make them a bit sparse, 
nri = lambda n,r: array(random(n) * r,dtype=int) # generate some random index
nw = sum([len(x) for x in w.nonzero()])
nu = sum([len(x) for x in u.nonzero()])
# A random set of words to 0 for all regions for all tasks
w[:,:,nri(W/2,W)] = 0
u[:,:,nri(U/2,U)] = 0


# now generate some random training data
X = np.random.random((N, R, U, W))
X[np.random.random((N, R, U, W)) < 0.9] = 0

# construct the response variable y = u X w + b
Xw = np.diagonal(
	np.tensordot(X,w,axes=([3],[2])),
	axis1=1, axis2=3
)
uXw_c = np.tensordot(u,Xw,axes=([2],[1]))
Y = np.diagonal(
		np.diagonal(
			uXw_c,
			axis1=1,axis2=3)
		,axis1=0,axis2=2) + b

# add some noise to make things more fun
Y += np.random.random(Y.shape)

def extract_days(*days):
	Yd = Y[days,:,:]
	Xd = X[days,:,:,:]
	Xud = ssp.csc_matrix(Xd.transpose([1,0,2,3]).reshape([R * len(days) * U, W]))
	Xwd = Xd.transpose([1,0,3,2]).reshape([R*len(days)*W,U])
	return Yd, Xud, Xwd

# Prepare the experiments, and let's go!
experiments = options.folds(options)
fold_n = 0
for fold in experiments:
	fold_dir = os.sep.join([output_dir,"fold_%d"%fold_n])
	os.makedirs(fold_dir)
	def save_epoch(epoch):
		epoch_file = os.sep.join([fold_dir,"epoch_%d"%epoch["epoch"]])
		logger.debug("Saving Epoch: %s"%epoch_file)
		epoch['Y_estimate'] = learner.predict(Xtest_wu,epoch['u_hat'],epoch['w_hat'],epoch['b_hat'] + epoch['Y_mean'],len(test))
		logger.debug("norm(b_hat): %s"%norm(epoch['b_hat']))
		epoch['Y_correct'] = Ytest
		epoch['Y_se'] = pow(epoch['Y_correct'] - epoch['Y_estimate'],2)
		epoch['Y_mse'] = mean(epoch['Y_se'])
		epoch['fold'] = fold
		logger.debug("Epoch_%d, Fold_%d, mse: %2.2f"%(epoch["epoch"],fold_n,epoch['Y_mse']))
		dump_compressed(epoch,epoch_file)
	
	learner = SparseRUWLearner(u_spams,w_spams, epoch_callback=save_epoch, epochs=options.nepochs,force_mean_center=options.force_mean_center)
	training = fold['training']
	test = fold['test']
	validation = fold['validation']
	logger.debug("Reading training data...")
	Ytrain,Xtrain_uw,Xtrain_wu = extract_days(*training)
	logger.debug("Reading test data...")
	Ytest,Xtest_uw,Xtest_wu = extract_days(*test)
	logger.debug("Learning...")
	learner.learn(Xtrain_uw,Xtrain_wu,Ytrain)
	fold_n += 1
