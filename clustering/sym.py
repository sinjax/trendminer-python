import sys
for line in sys.stdin:
  l=line.split()
  print l[0],l[1],l[2].strip()
  print l[1],l[0],l[2].strip()
