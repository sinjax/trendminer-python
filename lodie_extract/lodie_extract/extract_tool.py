from model import *
import argparse
import extract_single
from collections import defaultdict
import os

import extract_utils
from IPython import embed
import logging
import sys
import logging;logger = logging.getLogger("root")

parser = argparse.ArgumentParser(description='Extract annotations from files')
parser.add_argument('files', nargs='+')
parser.add_argument('-minuser', dest="minUserFreq",type=int)
parser.add_argument('-minterms', dest="minTermFreq",type=int, default=11)
parser.add_argument('-out', dest="out",type=str,default="out")
parser.add_argument('-fmt', dest="rdftype",type=str, default="nt")

options = parser.parse_args()

def ensure_dir(f):
    if not os.path.exists(f): os.makedirs(f)

ensure_dir(options.out)

userfreqs=defaultdict(int)
termmeta=defaultdict(dict)
records = []

for f in options.files: 
	logger.debug(str(f))
	records += [extract_single.extract_single(f,userfreqs,termmeta)]

users = extract_utils.filter_users(
	userfreqs,options.minUserFreq
)
tokenfreqs = defaultdict(int)
uids = Uids()
records = extract_utils.filter_records(
	records,tokenfreqs,users,uids=uids
)

graphs = extract_utils.records_to_graph(
	records,tokenfreqs,users,termmeta,
	minTokenCount=options.minTermFreq, uids=uids
)

for graph_name,graph in graphs.items():
	graph.serialize(
		destination=os.sep.join([options.out,"%s.%s"%(graph_name,options.rdftype)]),
		format=options.rdftype
	)