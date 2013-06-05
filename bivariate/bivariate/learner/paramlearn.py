import spams
from pylab import *

class LambdaSearch(object):
	"""
	Given a range of lambda values, a SpamsFunc instance
	an X and Y value and a cost function calculate the
	lambda which gives the best error

	spamsfunc - is a bivariate.learner.spamsfunc.SpamsFunctions 
			   which given an X and Y can learn a weighting
	errorfunc - is a bivariate.evaluator.lineareval.LinearEvaluator
				which given an expected Y and an X and calculated W
	"""
	def __init__(self, 
		spamsfunc, 
		errorfunc, 
	):
		super(LambdaSearch, self).__init__()
		self.spamsfunc = spamsfunc
		self.errorfunc = errorfunc
	
	def optimise(lambda_rng, x_parts, y_parts):
		min_err = None
		min_lambda = None
		for lmbda in lambda_rng:
			self.spamsfunc.params['lambda1'] = lmbda
			theta_new,beta = self.spamsfunc.call(x_parts.train, y_parts.train)
			err = self.errorfunc.evaluate(x_parts.val_param,y_parts.val_param,theta_new)
			if min_err is None or err < min_err:
				min_err = err
				min_lambda = lmbda
		return min_lambda



