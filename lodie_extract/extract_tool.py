
import argparse
import extract_single
from collections import defaultdict

import extract_utils
from IPython import embed
parser = argparse.ArgumentParser(description='Extract annotations from files')
parser.add_argument('files', nargs='+')
parser.add_argument('-minuser', dest="minUserFreq",type=int)

options = parser.parse_args()

userfreqs=defaultdict(int)
records = []
for f in options.files: records += [extract_single.extract_single(f,userfreqs)]

users = extract_utils.filter_users(userfreqs,options.minUserFreq)
tokenfreqs = defaultdict(int)
records = extract_utils.filter_records(records,tokenfreqs,users)

embed()