from scipy import io as sio
from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")
from experimentfolds import *
from optparse import OptionParser
from ..learner.batch.regionuserwordlearner import SparseRUWLearner,prep_wspams,prep_uspams,prep_w_graphbit
import cPickle as pickle
parser = OptionParser()
parser.add_option("-x", "--user-word-mat", dest="xroot",
                  help="The root of the user word matrices", metavar="FILE")
parser.add_option("-y", "--polls", dest="polls",metavar="FILE",
                  help="The polls tensor file")
parser.add_option("-f", "--nfolds", dest="nfolds",default=1,
                  help="Number of folds")
parser.add_option("-d", "--ndays", dest="ndays", default=235,
                  help="Number of days")
parser.add_option("-w", "--w-spams", dest="w_spams_file", default=None, metavar="FILE",
                  help="Number of days")
parser.add_option("-n", "--nthreads", dest="nthreads", default=-1,
                  help="Number of spams threads")
(options, args) = parser.parse_args()
experiments = prepare_folds(options.ndays,options.nfolds)
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

w_spams = prep_wspams(U,W,T,R,graphbit=w_spams_graphbit,lambda1=0.001)
u_spams = prep_uspams(lambda1=0.01)

w_spams.params['numThreads'] = int(options.nthreads)
u_spams.params['numThreads'] = int(options.nthreads)
learner = SparseRUWLearner(u_spams,w_spams)

for fold in experiments:
	training = fold['training']
	test = fold['test']
	validation = fold['validation']
	logger.debug("Reading training data...")
	Xtrain_uw,Xtrain_wu = read_split_userwordregion(options.xroot,cache=dataCache,*training)
	Ytrain = Y[training,:,:]
	logger.debug("Learning...")
	learner.learn(Xtrain_uw,Xtrain_wu,Ytrain)



