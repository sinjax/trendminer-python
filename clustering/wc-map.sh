#!/usr/bin/python
import json
import sys
import re
for line in sys.stdin:
  try:
    tweet=json.loads(line)
    tokenset= [x.lower() for x in tweet['analysis']['tokens']['all']]
    tokens=set(tokenset)
    for i in tokens:
      print i.lower(),'\t',1
  except:
    pass
