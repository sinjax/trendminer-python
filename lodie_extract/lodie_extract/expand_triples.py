import rdflib
from IPython import embed

DIRECT_CLASSES = """
	PREFIX owl: <http://www.w3.org/2002/07/owl#>
	SELECT ?date ?class (SUM(?count) AS ?aggregated)
	WHERE {
		?instance tm:word ?word .
		?instance tm:date ?date .
		?instance tm:count ?count.
		?word owl:Class ?class
	} 
	GROUP BY ?class
"""
format_map = {
	"owl": "xml"
}
def _find_format(inp):
	ret = inp.split(".")[-1]
	if ret in format_map:
		return format_map[ret]
	return ret;
def direct_classes(input,output,words={}):
	g = rdflib.Graph()
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	[g.load(file(inp,"r"),format=_find_format(inp)) for inp in input]
	expanded_results = g.query(DIRECT_CLASSES)
	expanded_graph = rdflib.ConjunctiveGraph()
	expanded_graph.bind("tm",TM)
	word_index = 0
	if len(words.values()) != 0: word_index = max(words.values()) + 1
	for binding in expanded_results.bindings:
		if not "class" in binding or not "user" in binding: continue
		if not binding['class'] in words:
			words[binding['class']] = word_index
			word_index+=1
		discussion = rdflib.BNode()
		expanded_graph.add((discussion,TM['word'],binding['class']))
		expanded_graph.add((discussion,TM['date'],binding['date']))
		expanded_graph.add((discussion,TM['count'],binding['aggregated']))
	return expanded_graph

ALL_CLASSES = """
	PREFIX owl: <http://www.w3.org/2002/07/owl#>
	prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
	SELECT ?user ?date ?class ?article_count (SUM(?count) AS ?aggregated)
	WHERE {
		?instance tm:word ?word .
		?instance tm:date ?date .
		?instance tm:count ?count.
		?instance tm:byuser ?user .
		?article_instance tm:byuser ?user.
		?article_instance tm:article_count ?article_count .
		?word owl:Class/rdfs:subClassOf* ?class .
	} 
	GROUP BY ?class ?user
"""


def all_classes(input,output,words={}):
	g = rdflib.Graph()
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	[g.load(file(inp,"r"),format=_find_format(inp)) for inp in input]
	expanded_results = g.query(ALL_CLASSES)
	expanded_graph = rdflib.ConjunctiveGraph()
	expanded_graph.bind("tm",TM)
	word_index = 0
	if len(words.values()) != 0: word_index = max(words.values()) + 1
	for binding in expanded_results.bindings:
		if not "class" in binding or not "user" in binding: continue
		if not binding['class'] in words:
			words[binding['class']] = word_index
			word_index+=1
		discussion = rdflib.BNode()
		expanded_graph.add((discussion,TM['word'],binding['class']))
		expanded_graph.add((discussion,TM['byuser'],binding['user']))
		expanded_graph.add((discussion,TM['date'],binding['date']))
		expanded_graph.add((discussion,TM['count'],binding['aggregated']))

		article_count = rdflib.BNode()
		expanded_graph.add((article_count,TM['byuser'],binding['user']))
		expanded_graph.add((article_count,TM['date'],binding['date']))
		expanded_graph.add((article_count,TM['article_count'],binding['article_count']))

	return expanded_graph