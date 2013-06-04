import bivariate.generator.randombigenerator as randomgen
from pylab import *

def test_randomge():
	gen = randomgen.RandomBiGen(noise=0.01)
	for x in range(10):
		x,y,w,u = gen.generate(include_wu=True)[0]
		assert (diag(u.T.dot(x).dot(w)) - y).sum() < 0.1
