from IPython import embed
from pylab import *
class LinearEvaluator(object):
	"""Evaluate a bilinear learner"""
	def __init__(self):
		super(LinearEvaluator, self).__init__()
	
	def evaluate(self,X,Y,theta,bias=None):
		pass

class SquareEval(LinearEvaluator):
	"""sum of the square difference between y and x . theta + bias"""
	def __init__(self):
		super(SquareEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		total = 0;
		dotproduct = X.dot(theta)
		if bias is not None: dotproduct += bias
		diff = Y - dotproduct
		diff = diff[~np.isnan(diff)]
		total = pow(diff[~np.isnan(diff)],2).sum()
		dotproduct[np.isnan(Y)] = 0
		diffmat = ((Y - dotproduct)[~np.isnan(Y)])
		diffmat = diffmat.reshape(Y.shape[1],Y.shape[0]/Y.shape[1]).T
		diffmatsse = pow(diffmat,2).sum(axis=0)
		return {
			"totalsse":total,
			"diff":diffmat,
			"diffsse":diffmatsse,
			"dotproduct":dotproduct
		}

class MeanEval(SquareEval):
	"""The SquareEval divided by the total number of Ys (i.e. tasks and days)
	being estimated"""
	def __init__(self):
		super(MeanEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		ssed = super(MeanEval,self).evaluate(X,Y,theta,bias)
		ndays = Y.shape[0]/Y.shape[1]
		ssed["totalsse"] = ssed["totalsse"]/Y[~np.isnan(Y)].size
		ssed["diff"] = ssed["diff"]/ndays
		ssed["diffsse"] = ssed["diffsse"]/ndays
		return ssed

class RootMeanEval(MeanEval):
	"""The square root of the MeanEval"""
	def __init__(self):
		super(RootMeanEval, self).__init__()

	def evaluate(self,X,Y,theta,bias=None):
		ssed = super(RootMeanEval,self).evaluate(X,Y,theta,bias)
		ssed["totalsse"] = sqrt(ssed["totalsse"])
		ssed["diffsse"] = sqrt(ssed["diffsse"])
		return ssed
		
		
		