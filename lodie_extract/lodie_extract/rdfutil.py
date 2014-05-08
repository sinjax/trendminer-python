import rdflib
def load_graph(f):
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	g = rdflib.ConjunctiveGraph()
	g.load(f,format=f.split(".")[1])
	g.bind("tm",TM)
	return g
def load_index(graphf,query):
	graph = load_graph(graphf)
	results = graph.query(query)
	
	index = {}
	for r in results: index[r[0].toPython()] = r[1].toPython()
	n = max(index.values()) + 1
	return n, index

def construct_word_graph(words):
	g = rdflib.Graph()
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	g.bind("tm",TM)
	for word,index in words.items():
		g.add((word,TM['index'],rdflib.Literal(index,datatype=rdflib.XSD.integer)))
	return g