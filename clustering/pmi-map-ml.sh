#!/usr/bin/python
import sys
import random
import re
import json

fw=open(sys.argv[1],"r")
ids={}
for line in fw:
  l=line.split()
  w=l[1]
  ids[w]=l[0]
fw.close()
for line in sys.stdin:
    try:
      tweet=json.loads(line)
      tokenset= [i.lower().strip('"') for i in tweet['analysis']['tokens']['all']]
      if not tokenset:
        continue
      tokenset=list(set(tokenset))
      tokenset.sort()
      L=len(tokenset)
      for i in range(0,L):
        for j in range(i+1,L):
          if tokenset[i]!=tokenset[j]:
            w1=tokenset[i].strip('"')
            w2=tokenset[j].strip('"')
            try:
              id1=ids[w1]
              id2=ids[w2]
            except KeyError:
              continue    
            print id1,id2,'\t',1
    except:
      pass
