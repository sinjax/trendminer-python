# coding=utf-8
from textprocess import process
import tldextract
import re
from IPython import embed
from collections import defaultdict
annots = {
	"officeowl":"http://dbpedia.org/ontology/office", 
	"partyowl":"http://dbpedia.org/ontology/PoliticalParty", 
	"birthplaceowl":"http://dbpedia.org/ontology/birthPlace", 
	"countryowl":"http://dbpedia.org/ontology/Country", 
	"leadernameowl":"http://dbpedia.org/ontology/leaderName", 
	"birthcountry":"http://dbpedia.org/ontology/birthPlace", 
	"officeprop":"http://dbpedia.org/property/office",
	"partyprop":"http://dbpedia.org/property/party", 
	"countryprop":"http://dbpedia.org/property/country"
}

ANNIE_annots = {
	"Person" : "http://dbpedia.org/ontology/Person",
	"Location" : "http://dbpedia.org/ontology/Place",
	"Organization" : "http://dbpedia.org/ontology/Organisation",
}

ANNIE = "http://dbpedia.org/resource"

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

def find_one_ANNIE(par, annietype,meta=defaultdict(dict)):
	annots = []
	for pers in par.find_all(annietype):
		
		end_name = "_".join(pers.stripped_strings)
		if len(end_name) == 0: continue
		name = "annotated:" + end_name
		update_meta(meta,name,uri="/".join([ANNIE,end_name]),type=ANNIE_annots[annietype])
		annots.append(name)
	
	return annots

def find_ANNIE(par,meta=defaultdict(dict)):
	annots = []
	annots.extend(find_one_ANNIE(par, "Person",meta))
	annots.extend(find_one_ANNIE(par, "Location",meta))
	annots.extend(find_one_ANNIE(par, "Organization",meta))
	return annots

def find_Mentions(par,meta=defaultdict(dict)):
	annotations = []
	for mention in par.find_all("Mention"):
		name = "annotated:" + mention["inst"].replace("http://dbpedia.org/resource/", "")
		annotations.append(name)
		if "class" in mention.attrs:
			update_meta(meta,name,uri=mention["inst"],type=mention["class"])
		else:
			update_meta(meta,name,uri=mention["inst"],type=None)
		for annot in annots:
			if annot in mention.attrs:
				props = mention[annot].split(",")
				names = []
				for p in props:
					p = p.strip()
					if p:
						p = p.replace(" ", "_")
						if "http" not in p:
							p = "http://dbpedia.org/resource/" + p
						name = "annotated:" + p.replace("http://dbpedia.org/resource/", "")
						if name:
							names += [name]
							update_meta(meta,name,uri=p,type=annots[annot])
				annotations.extend(names)
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

def find_annotations(par,meta=defaultdict(dict)):
	annotations = []
	annotations.extend(find_ANNIE(par,meta))
	annotations.extend(find_Mentions(par,meta))

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

def update_meta(meta,name,uri=None,type=None):
	if not name in meta:
		meta[name]['uri'] = uri
		if type:
			meta[name]['type'] = type