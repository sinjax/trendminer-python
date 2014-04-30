# coding=utf-8
import logging
logging.basicConfig()
from os.path import *
from dateutil.parser import parse
from bs4 import BeautifulSoup
import codecs
from model import *
from annutils import *
from collections import defaultdict



def extract_single(fil,userfreqs=defaultdict(int),termmeta=defaultdict(dict)):

	date = basename(fil)[:-8]
	curdate = parse(date)
	

	soup = BeautifulSoup(codecs.open(fil, "r", encoding='utf-8'), 'xml')

	## one or two pages have embedded scripts that muck things up
	## removing them doesn't remove any readable text on page.
	for script_block in soup.find_all("script"):
		script_block.clear()

	data = soup.find("div", "newsbody")

	curRecord = Record(curdate)
	curUsersCounted = set()

	curtext = []
	annotations=[]

	for par in data("p"):
		links = par.findAll("a")
		if links != []:
			users = set()
			for link in links:
				domain = extract_valid_domain(link)
				if domain:
					users.add(domain)
					link.clear()

			txt = list(par.stripped_strings)
			if users:

				## find mentions, properties etc
				annotations.extend(find_annotations(par,meta=termmeta))
				update_counts(userfreqs, [u for u in users if u not in curUsersCounted])
				curUsersCounted.union(users)
				curRecord.items.append(Item(annotations, list(users)))
				annotations=[]
				#break
		else:
			txt = list(par.stripped_strings)
			if not txt: continue
			## paragraph ends with the text 'No Link', remove No Link text
			## end current story
			if len(txt)==2 and txt[0]==u"No" and txt[1]==u"link":
				annotations.extend(find_annotations(par,meta=termmeta))
				update_counts(userfreqs, [u"No Source"] if u"No Source" not in curUsersCounted else [])
				curUsersCounted.add(u"No Source")
				curRecord.items.append(Item(annotations, [u"No Source"]))
				annotations = []
				#break
			else:
				annotations.extend(find_annotations(par,meta=termmeta))
	return curRecord