import bivariate.learner.spamsfunc as sf
from pylab import *
from scipy import sparse as ssp
import bivariate.evaluator.bilineareval as sse

def test_fistaflat():
	params = {
		"loss":"square",
		"lambda1": 0.1, 
		"it0": 3,
		"regul":"l2",
		"max_it": 10,
		"intercept" : True
	}
	ff = sf.FistaFlat(params=params)
	user = 10
	word = 5
	day = 3
	tasks = 1
	x = rand(day,word)
	y = rand(day,tasks)
	w,b = ff.call(ssp.csc_matrix(x),np.asfortranarray(y))
def test_fistatree():
	params = {
		"loss":"square",
		"lambda1": 0.1, 
		"it0": 3,
		"regul":"l2",
		"max_it": 10,
		"intercept" : True
	}

	user = 10
	word = 5
	day = 3
	tasks = 1
	own_variables =  np.array([0],dtype=np.int32)
	N_own_variables =  np.array([word],dtype=np.int32)
	eta_g = np.array([1],dtype=np.float64)
	tree = {
		'eta_g': eta_g,
		'groups' : ssp.csc_matrix((1,1),dtype=bool),
		'own_variables' :own_variables,
        'N_own_variables' : N_own_variables
    }

	ff = sf.FistaTree(tree=tree,params=params)
	x = rand(day,word)
	y = rand(day,tasks)
	w,b = ff.call(ssp.csc_matrix(x),np.asfortranarray(y))
