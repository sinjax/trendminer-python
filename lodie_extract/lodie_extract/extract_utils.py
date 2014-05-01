from model import *
from annutils import *
import rdflib
def filter_users(userfreqs,minUserFreq):
	users = set([user for (user, freq) in userfreqs.items() if freq >= minUserFreq])
	return users
def filter_records(records,tokenfreqs,users,uids=Uids()):
	
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

	return filtered_records
def find_oldest_date(records):
	oldest_date = None
	for rec in records:
		if not oldest_date or oldest_date > rec.day:
			oldest_date = rec.day
	return oldest_date
def records_to_graph(records, tokenfreqs, users, termmeta, minTokenCount=11,uids=Uids()):
	## find top N tokens
	tokens = sorted(
		tokenfreqs.items(), reverse=True, key=lambda (a,b): b
	)
	tokens = [
		word for word, count in tokens if count>minTokenCount
	]
	#tokens = sorted(tokenfreqs.items(), reverse=True, key=lambda (a,b): b)[:N]
	#tokens = [word for word, count in tokens]
	tokendict = dict(zip(tokens, range(len(tokens))))

	records.sort(key=lambda a: a.day)

	def filtered_frequency(words):
		data = defaultdict(int)
		for word in words:
			if word in tokendict:
				data[word] += 1
		return data

	## date index
	days = []
	oldest_date = find_oldest_date(records)

	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	ret_dict = {}
	user_rev = dict([(y,x) for (x,y) in uids.users.items()])
	for rec in records:
		data_graph = rdflib.ConjunctiveGraph()
		data_graph.bind("tm",TM)
		## for date index
		daystring = rec.day.date().isoformat()
		cur_date = rec.day
		rec.day = (rec.day - oldest_date).days

		days.append((rec.day, daystring))

		curdata = defaultdict(lambda: [])
		curcounts = defaultdict(int)
		for article in rec.items:
			for user in article.users:
				curdata[user] += article.txt
				curcounts[user] += 1
		for user in curdata:
			certain_date = rdflib.BNode()
			data_graph.add((certain_date,TM['date'], rdflib.Literal(cur_date,datatype=rdflib.XSD.date)))
			data_graph.add((certain_date,TM['byuser'], rdflib.Literal(user_rev[user],datatype=rdflib.XSD.string)))
			data_graph.add((certain_date,TM['count'], rdflib.Literal(curcounts[user],datatype=rdflib.XSD.integer)))
			for word, freq in filtered_frequency(curdata[user]).items():
				discussion = rdflib.BNode()
				data_graph.add((discussion,TM['date'], rdflib.Literal(cur_date,datatype=rdflib.XSD.date)))
				data_graph.add((discussion,TM['byuser'], rdflib.Literal(user_rev[user],datatype=rdflib.XSD.string)))
				data_graph.add((discussion,TM['word'], rdflib.URIRef(termmeta[word]['uri'])))
				data_graph.add((discussion,TM['count'], rdflib.Literal(freq,datatype=rdflib.XSD.integer)))
		ret_dict["data_" + daystring] = data_graph

	word_graph = rdflib.ConjunctiveGraph()
	word_graph.bind("tm",TM)
	for word, index in sorted(tokendict.items(), key=lambda x: x[1]):
		word_uri = rdflib.URIRef(termmeta[word]['uri'])
		if 'type' in termmeta[word]:
			clz = rdflib.URIRef(termmeta[word]['type'])
			word_graph.add((word_uri,rdflib.OWL.Class,clz))

		word_graph.add((word_uri,TM['index'],rdflib.Literal(index,datatype=rdflib.XSD.integer)))
	ret_dict["word"] = word_graph

	user_graph = rdflib.ConjunctiveGraph()
	user_graph.bind("tm",TM)
	for uname, index in sorted(uids.users.items(), key=lambda x: x[1]):
		user_node = rdflib.BNode()
		user_graph.add((user_node,TM['isuser'],rdflib.Literal(uname,datatype=rdflib.XSD.string)))
		user_graph.add((user_node,TM['index'],rdflib.Literal(index,datatype=rdflib.XSD.integer)))
	ret_dict["user"] = user_graph

	date_graph = rdflib.ConjunctiveGraph()
	date_graph.bind("tm",TM)
	for index, date in sorted(days):
		date_node = rdflib.BNode()
		date_graph.add((date_node,TM['isdate'],rdflib.Literal(date,datatype=rdflib.XSD.string)))
		date_graph.add((date_node,TM['index'],rdflib.Literal(index,datatype=rdflib.XSD.integer)))

	ret_dict["date"] = date_graph

	return ret_dict
	