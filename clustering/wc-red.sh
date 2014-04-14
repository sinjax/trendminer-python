#!/usr/bin/python
import sys
prevWord=""
valueTotal=0
for line in sys.stdin:
	(word,values)=line.split('\t')
	value=int(values.strip())
	if word==prevWord or prevWord=='':
		valueTotal=valueTotal+value
		prevWord=word
	else:
		print prevWord,'\t',valueTotal
        	prevWord=word
		valueTotal=value
print prevWord,'\t',valueTotal
