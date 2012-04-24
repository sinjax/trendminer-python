import sys

result_files = ["results_con","results_lab","results_lib"]

# open results files (con,lab,lib)
results = [open(fname,'w') for fname in result_files]

# go through each output file
for n in range(1,4):
	fname = "tweets_pre{0}.list.out".format(n)
	with open(fname) as f:
		while 1:
			# read 3 consecutive lines (con,lab,lib)
			day = [f.readline() for n in range(3)]
			if day[0] == "":
				break
			# write each line to the appropriate file (con,lab,lib)
			for i in range(3):
				results[i].write(day[i])

# close reults files
for f in results:
	f.close()

