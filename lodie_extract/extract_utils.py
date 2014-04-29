from model import *
from annutils import *
def filter_users(userfreqs,minUserFreq):
	users = set([user for (user, freq) in userfreqs.items() if freq >= minUserFreq])
	return users
def filter_records(records,tokenfreqs,users):
	uids = Uids()
	## filter out low-frequency news sources,
	## remove news items that have no users
	filtered_records = []
	for rec in records:
	    newItems = []
	    for item in rec.items:
	        newusers = []
	        for user in item.users:
	            if user in users:
	                newusers.append(uids.getUid(user))
	        if newusers:
	            newItems.append(Item(item.txt, newusers))
	            update_counts(tokenfreqs, item.txt)
	    newRec = Record(rec.day)
	    newRec.items = newItems
	    filtered_records.append(newRec)
	return records