from pylab import *
from scipy import sparse as ssp
from scipy import io as sio
from IPython import embed
import sys
import rdflib
import logging;logger = logging.getLogger("root")
from ..rdfutil import *
from collections import defaultdict



DATE_QUERY = """
	SELECT ?key ?index WHERE {
		?a tm:isdate ?key.
		?a tm:index ?index.
	}
"""

WORD_QUERY = """
	SELECT ?key ?index WHERE {
		?key tm:index ?index.
	}
"""

USER_QUERY = """
	SELECT ?key ?index WHERE {
		?a tm:isuser ?key.
		?a tm:index ?index.
	}
"""

DATA_QUERY = """
	SELECT ?user ?date ?word ?count WHERE {
    	?post tm:byuser ?user .
    	?post tm:count ?count .
    	?post tm:date ?date .
    	?post tm:word ?word .
    }
"""

DATA_QUERY = """
	SELECT ?user ?date ?word ?count WHERE {
    	?post tm:byuser ?user .
    	?post tm:count ?count .
    	?post tm:date ?date .
    	?post tm:word ?word .
    }
"""
DATA_AGGR_QUERY = """
	SELECT ?user ?date ?word ?count WHERE {
    	?post tm:byuser ?user .
    	?post tm:article_count ?count .
    	?post tm:date ?date .
    }
"""

def create_sparse(data_graphs,**xargs):
	# load the day index graph

	ndays,days_index =  load_index(xargs["dategraph"],DATE_QUERY)
	nwords,word_index = load_index(xargs["wordgraph"],WORD_QUERY)
	nusers,user_index = load_index(xargs["usergraph"],USER_QUERY)
	logger.debug("Creating both matrices")
	query_entries = defaultdict(list)
	queries = [DATA_QUERY,DATA_AGGR_QUERY]
	for data_graph in data_graphs:
		logger.debug("Loading and querying: " + data_graph)
		data = load_graph(data_graph)
		for query in queries:
			embed()
			entries = query_entries[query]
			results = data.query(query)
			for result in results:
				user = user_index[result['user'].toPython()]
				day = days_index[result['date'].toPython().strftime("%Y-%m-%d")]
				if result['word'] == None:
					word = 0
				else:
					word = word_index[result['word'].toPython()]
				count = result['count'].toPython()
				entries += [[int(day),int(user),int(word),int(count)]]
		asddsasa
	results = []
	for query in queries:
		entries = query_entries[query]
		if query == DATA_QUERY:
			M = (nusers) * (ndays)
			N = (nwords)
		else:
			M = (nusers) * (ndays)
			N = 1
		logger.debug("Done reading, creating matrix: %d x %d "%(M,N))
		out = ssp.dok_matrix((M,N))
		mkey = (lambda d,u: (d * nusers) + u)
		for entry in entries:
			day,user,word,count = entry
			out[mkey(day,user),word] = count
		out = getattr(out,"to%s"%xargs['type'])()
		results += [out]
	return results
	
