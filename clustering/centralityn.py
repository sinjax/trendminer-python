from IPython import embed
def centralityn(c,M,P):
	a=[];
	for k in range(len(c)):
		val=0;
		for kk in range(len(c)):
			val=val+M[c[k],c[kk]];
		a += [val];
	wordsix=P[c];
	b=a/sum(a);
	return b,wordsix
