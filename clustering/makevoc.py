import sys
c=0
for line in sys.stdin:
  l=line.strip().split()
  c=c+1  
  print c,l[0]
