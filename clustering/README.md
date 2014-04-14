# Trendminer python clustering repo

This is a pure python port of [Trendminer matlab clustering] (https://github.com/danielpreotiuc/trendminer-clustering)
## Description

Spectral clustering work done for D3.2.1 of the [Trendminer project] (http://www.trendminer-project.eu/). This builds hard clusters of words that co-occur in the same tweets. More information about the algorithm is available in the [deliverable] (http://www.trendminer-project.eu/).

## Installation

First install all the requirements using pip

	pip install -r requirments.txt

The project requires pylab (numpy, scipy etc.), cherrypy and sckit-learn

The code is available as a collection of Python scripts. You need to have both installed. The script that performs the entire sequence of operations is written in shell and tested under Ubuntu.

## Quick start

	./run file voc.threshold npmi.threshold k no.clusters

file - input file of tweets. One tweet in Json format per line.  This must be preprocessed using the [Trendminer Preprocessing pipeline] (https://github.com/sinjax/trendminer-java), needing tokenisation.

voc.threshold - vocabulary cut-off

npmi.threshold - NPMI cut-off

k - k of the mutual K nearest-neighbour graph for spectral clustering

no.clusters - number of clusters 

**Example:**
	
	./run.sh sep5 1 0.1 30 30

sep5 - sample tweets file processed for language and tokenized using the Trendminer pipeline. Fields are restricted to only text due for privacy. It largely consists of tweets written on 5 September 2013 in Austria, filtered for german and relating to politics.

## Webservice

To run, start first the webservice script:

	python wsscl.py cluster.file cluster.label.file

cluster.file is the file with the cluster asssignments in the format (word clusterid importance) , same as the output of the run script. Default is cl.

cluster.label.file is a file with the cluster coherence and labels given as one (coherence label) pair each line. Default is cl.lab or cluster.file.lab if the first parameter is given.

Then use curl to interact with the service. There are 3 endpoints, all returning JSON objects:

#### GET /cluster?c=cid&t=topw

returns the cluster with cid as its label, coherence score and the top topw words and their importance for the topic.

#### GET /words?w=word

returns the id of the cluster where the word belongs.

#### POST /topics?t=notop&coh=c

receives a tweet file, 1 json/line and returns the top notop topics in the file and their importance score. coh filters to include only topics with at least c coherence rating (default 3).

**Examples:**	

	python wsscl.py cl-sora-200w-ncen cl-sora-200-labels

	curl -X GET  http://localhost:8080/cluster?c=199\&t=5
	
	cat mar13 | curl -i -k -H "Content-Type: application/json" -H "Accept: application/json" -X POST --data-binary @- http://localhost:8080/topics?t=5

## Scripts

This is a description of each script. They are all needed to be ran in the order given by the run.sh script but can also be useful individually.

#### dedup.py

Performs tweet deduplication. Uses a very aggressive method that removes all tweets that are detected as being retweets or that have the same first 5-6 word tokens.

	cat file | python dedup.py > file-dedup

#### wc-map.sh, wc-red.sh

Performs word count on all tokens in a tokenised tweet file. Written in Map-Reduce 2-step style.

	cat file | python wc-map.sh | sort | python wc-red.sh > wc.file

#### wc-filter.py

Filters the word count file by imposing a minimum count value.

	cat wc.file | python wc-filter.py threshold > wc.file.threshold

#### makevoc.py

	cat wc.file | python makevoc.py > dict.file

Gets as input a (word frequency) file and outputs a (id word) index.

#### wc-map-dict.sh

	cat file | python wc-map-dict.sh dict.file | sort | python wc-red.sh > wc.file

Performs word count on the tokens that exist in the vocabulary file. Written in MapReduce style.

#### pmi-map-ml.sh

        cat file | python pmi-mal-ml.sh dict.file | sort | python wc-red.sh > wco.file

Performs word co-occurrence count on the tokens that exist in the vocabulary file. Written in MapReduce style. Output pairs are ordered alphabetically.

#### compute-npmi.py

	python compute-npmi.py wc.file wco.file > npmi.file

Computes the NPMI values of word pairs using the word occurrence and co-occurrence counts.

#### truncate.py

	cat npmi.file | python truncate.py threshold > npmi.file.threshold

Truncates a NPMI values file to a threshold.

#### sym.py

	cat npmi.file | python sym.py > npmi.file.symmetric

Makes an NPMI file symmetric.

#### spectral.py

	spectral(file,npmi.file,dict.file,k,no.clusters)

Performs spectral clustering using the tweet file, npmi file, dictionary.  K and no.clusters are parameters of the spectral clustering algorithm.

Spectral clustering is currently performed using the code from the [Scikit-Learn] (http://scikit-learn.org/).
