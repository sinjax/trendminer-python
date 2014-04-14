import sys
THRESHOLD=float(sys.argv[1])
for line in sys.stdin:
  l=line.split()
  if float(l[2])>THRESHOLD:
    print line,
