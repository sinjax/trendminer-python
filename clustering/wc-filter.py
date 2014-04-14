import sys
import re
THRESHOLD=int(sys.argv[1])
for line in sys.stdin:
  l=line.split()
  word=l[0].strip()
  if word[0]=="#" or word[0].isalpha():
    try:
      if int(l[1])>THRESHOLD:
        print line,
    except:
      pass
