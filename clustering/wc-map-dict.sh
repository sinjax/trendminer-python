#!/usr/bin/python
import json
import sys
import re
fw=open(sys.argv[1],"r")
ids={}
for line in fw:
  l=line.split()
  w=l[1].lower()
  ids[w]=l[0]
fw.close()
for line in sys.stdin:
  try:
    tweet=json.loads(line)
    tokenset= [x.lower() for x in tweet['analysis']['tokens']['all']]
    tokens=set(tokenset)
    for i in tokens:
      try:
        ii=ids[i.lower()]
        print ii,'\t',1
      except:
        continue
  except:
    pass
