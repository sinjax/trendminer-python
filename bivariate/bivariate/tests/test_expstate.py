import bivariate.experiment.expstate as es
from pylab import *
def test_expstate():
	es.exp("the_experiment")
	es.state("default")["cheese"] = 2
	es.state()["blah"] = rand(100,20)
	es.flush()
	es.state("next")["cheese"] = 3
	es.state()["blah"] = rand(100,20)
	es.flush()
	print es.load_states()