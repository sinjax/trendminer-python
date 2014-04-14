import math
import sys
f=open(sys.argv[1],"r")
count={}
NO_WORDS=0
for line in f:
  l=line.split('\t')
  w=int(l[0])
  c=int(l[1])
  count[w]=c
  NO_WORDS=NO_WORDS+c
f.close()
f=open(sys.argv[2],"r")
NO_PAIRS=0
for line in f:
  l=line.split('\t')
  c=int(l[1])
  NO_PAIRS=NO_PAIRS+c
f.close()
f=open(sys.argv[2],"r")
for line in f:
  li=line.split('\t')
  coocc=int(li[1])
  l=li[0].split()
  c1=count[int(l[0])]
  c2=count[int(l[1])]
  if c1==coocc or c2==coocc:
    continue
  if c1>10 and c2>10:
    p1=math.log(float(coocc)*float(NO_WORDS)*float(NO_WORDS)/(c1*c2*float(NO_PAIRS)))
    p2=math.log(float(coocc)/float(NO_PAIRS))*(-1.0)
    print "%s %s %0.10f" % (l[0],l[1],p1/p2)
f.close()
