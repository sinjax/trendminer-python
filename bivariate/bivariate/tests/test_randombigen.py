import bivariate.generator.randombigenerator as randomgen
from pylab import *

def test_randomge():
	gen = randomgen.RandomBiGen(noise=0.01)
	for x in range(10):
		# print gen.generate(include_wu=True)
		x,y,w,u = gen.generate(include_wu=True)
		diff = diag(u.T.dot(x).dot(w)) - y
		assert abs(diff).sum() < 0.1

def test_bigbias():
	gen = randomgen.RandomBiGen(noise=0.01,brng=(100,1000))
	for x in range(10):
		x,y,w,u,bias = gen.generate(include_wu=True,include_bias=True)
		estim = diag(u.T.dot(x).dot(w)) + bias
		diff = (estim - y)
		assert abs((diff).sum()) < 0.1