import bivariate.learner.spamsfunc as sf
from pylab import *
from scipy import sparse as ssp

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
	ff.call(ssp.csc_matrix(x),np.asfortranarray(y))
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
	
	tree = {
		'eta_g': eta_g,
		'groups' : groups,
		'own_variables' : own_variables,
        'N_own_variables' : N_own_variables
    }

	ff = sf.FistaTree(tree=None,params=params)
	x = rand(day,word)
	y = rand(day,tasks)
	ff.call(ssp.csc_matrix(x),np.asfortranarray(y))