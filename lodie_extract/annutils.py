# coding=utf-8
from textprocess import process
import tldextract
import re
from IPython import embed
annots = ["officeowl", "partyowl", "birthplaceowl", "countryowl", "leadernameowl", "birthcountry", "officeprop", "partyprop", "countryprop"]

extract = tldextract.TLDExtract(suffix_list_url=None)
ipaddress = re.compile(r"[0-9]+(\.[0-9]+){3}$")
## extract valid data from some odd links in the text
def extract_valid_domain(link):
	if link.has_attr("href"):
		link = link["href"]
		## deal with redirector
		lastredir = link.rfind('redir.asp?URL=')
		if lastredir > 0:
			link = link[lastredir + 14:]
		if link.startswith("http"):
			tldtuple = extract(link)
			## don't want to treat ip-addresses as sources
			if not ipaddress.match(tldtuple.domain):
				## could consider using only domain, without suffix
				return tldtuple.domain + "." + tldtuple.suffix
	return None
total = 0.0
mentionCount = 0.0
def find_one_ANNIE(par, annietype):
	annots = []
	global total, mentionCount
	for pers in par.find_all(annietype):
		total += 1
		#pers.stripped_strings = list(pers.stripped_strings)
		annots.append("annotated:" + "_".join(pers.stripped_strings))
		mentions = pers.find_all("Mention")
		if len(mentions) != 0:
			mentionCount += 1
	if total!= 0:
		print mentionCount/total
	return annots

def find_ANNIE(par):
	annots = []
	annots.extend(find_one_ANNIE(par, "Person"))
	annots.extend(find_one_ANNIE(par, "Location"))
	annots.extend(find_one_ANNIE(par, "Organization"))
	return annots

def find_Mentions(par):
	annotations = []
	for mention in par.find_all("Mention"):
		annotations.append("annotated:" + mention["inst"].replace("http://dbpedia.org/resource/", ""))
		for annot in annots:
			if annot in mention.attrs:
				props = mention[annot].replace("http://dbpedia.org/resource/", "").replace(" ", "_").split(",")
				props = ["annotated:" + p for p in props if p]
				annotations.extend(props)
	return annotations

def find_POS(par):
	POStags=[]
	annotations = []
	for tok in par.find_all("Token"):
		x = process([tok["string"]])
		if x:
		   xx=x[0]+'/'+tok["category"]
		   POStags.append(xx)
	return POStags

def find_annotations(par):
	annotations = []
	annotations.extend(find_ANNIE(par))
	annotations.extend(find_Mentions(par))

	return annotations
def find_all(par):
	## find only mentions for now, could potentially filter
	## mentions or specialize them by types later
	annotations = []
	annotations.extend(find_ANNIE(par))
	annotations.extend(find_Mentions(par))
	annotations.extend(find_POS(par))
	
	return annotations

def update_counts(counts, words):
	for word in words:
		counts[word] += 1