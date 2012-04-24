import sys, json

def load_keywords(kw_file):
	kw_set = set()
	with open(kw_file) as f:
		for kw in f:
			kw_set.add(kw.strip())
	return kw_set

def get_PN_tweet(tokens):
	# count the number of positive words and negative words
	num_positive = 0
	num_negative = 0
	for tok in tokens:	# count up the number of positive and negative words in this tweet
		if tok in p_keywords:
			num_positive += 1
		if tok in n_keywords:
			num_negative += 1
	return (num_positive,num_negative)

def update_counts(kws,tokens,pn_count):
	# update counts for different keywords in countP and countN
	cp = 0
	cn = 0
	for tok in tokens:
		if tok in kws:
			cp += pn_count[0]
			cn += pn_count[1]
	return (cp,cn)


BASE = "/corpora/twitter/analysis/"

# avoid divide-by-zero if negatives is 0 
eps = 1

# read the keywords for each list (conservative, labour, liberal)
kw_con = load_keywords("/home/jingli/tw/code/keywords_con")
kw_lab = load_keywords("/home/jingli/tw/code/keywords_lab")
kw_lib = load_keywords("/home/jingli/tw/code/keywords_lib")

# read positive and negative keyword lists
p_keywords = load_keywords("/corpora/polls-tweets/positive")
n_keywords = load_keywords("/corpora/polls-tweets/negative")


# take a list of tweet filenames from the given file
if len(sys.argv) != 2:
	sys.stderr.write("Usage: mapper <tweets_list_file>\n")
	sys.exit(1)

tweets_list = sys.argv[1]	# the first parameter
out_filename = tweets_list+".out"
output_file = open(out_filename,'w')

with open(tweets_list) as tlist:
	for tweet_file in tlist:
		# set up positive count and negative counts for keywords
		count_con = (0,0)		# (pos,neg)
		count_lab = (0,0)
		count_lib = (0,0)
		# for each of those files
		fname = tweet_file.strip()
		with open(BASE+fname) as f:
			# for each line in the current file
			for line in f:
				try:
					# load the tweet
					tweet = json.loads(line)		
                                        # get the tokens from the tweet and change them into lowercase
					tokens = [t.lower() for t in  tweet['analysis']['tokens']['unprotected']]
					# get counts for pos & neg words in this tweet
					pn_count = get_PN_tweet(tokens)
					# update counts for each list
					##con
					(cp,cn,frq) = update_counts(kw_con,tokens,pn_count)
					count_con = (count_con[0]+cp,count_con[1]+cn)
					##lab
					(cp,cn,frq) = update_counts(kw_lab,tokens,pn_count)
					count_lab = (count_lab[0]+cp,count_lab[1]+cn)
					##lib
					(cp,cn,frq) = update_counts(kw_lib,tokens,pn_count)
					count_lib = (count_lib[0]+cp,count_lib[1]+cn)
				except:
					continue

			# print keyword positives, negatives, and ratio for that day
			# format: day [con|lab|lib] total_pos total_neg ratio
			numP = count_con[0]; numN = count_con[1]; ratio = (float(numP)+eps)/(float(numN)+eps)
			output_file.write("{0} con {1} {2} {3:.3}\n".format(fname,numP,numN,ratio))

			numP = count_lab[0]; numN = count_lab[1]; ratio = (float(numP)+eps)/(float(numN)+eps)
			output_file.write("{0} lab {1} {2} {3:.3}\n".format(fname,numP,numN,ratio))

			numP = count_lib[0]; numN = count_lib[1]; ratio = (float(numP)+eps)/(float(numN)+eps)
			output_file.write("{0} lib {1} {2} {3:.3}\n".format(fname,numP,numN,ratio))

output_file.close()
#sys.stdout.write(out_filename+"\n")
