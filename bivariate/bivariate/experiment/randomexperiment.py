import bivariate.generator.randombigenerator as randomgen
import bivariate.crossvalidation.timeseriescv as tscv
from bivariate.learner.spamsfunc import *
from bivariate.learner.partbatchbivariate import BatchBivariateLearner
from pylab import *
import os
from IPython import embed
import logging;logger = logging.getLogger("root")
import scipy.sparse as ssp
import bivariate.experiment.expstate as es

u_lambdas = np.arange(0.1,1,0.1)
w_lambdas = np.arange(0.1,2,0.1)
w_spams = FistaFlat(**{
	"intercept": True,
	"loss":"square",
	"regul":"elastic-net",
	"it0":10,
	"max_it":1000,
	"lambda2":0.5,
	"lambda1":0.3
})
u_spams = FistaFlat(**{
	"intercept": True,
	"loss":"square",
	"regul":"elastic-net",
	"max_it":1000,
	"lambda2":0.5,
	"lambda1":0.3
})

es.exp("randomExp",fake=False)
es.state("random")
gen = randomgen.RandomBiGen(noise=0.01,brng=(100,1000),ntasks=1,wu_sparcity=0.6,wrng=(2,3),urng=(2,3),nusers=100,nwords=400)
x,y = gen.generate(n=1000)
x = ssp.csc_matrix(vstack(x).T)
y = array(y)

fold = [f for f in tscv.tsfi(y.shape[0],ntest=100,ntraining=900)][0]
Xparts,Yparts = BatchBivariateLearner.XYparts(fold,x,y)

learner = BatchBivariateLearner(w_spams,u_spams,bivar_max_it=10)
learner.process(Yparts.train_all,Xparts.train_all,tests={"test":(Xparts.test,Yparts.test)})

print learner.w.todense()
print gen._w
print learner.u.todense()
print gen._u
print learner.w_bias
print learner.u_bias
print gen._bias
embed()