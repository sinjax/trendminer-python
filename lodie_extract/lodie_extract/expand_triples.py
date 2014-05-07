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
def direct_classes(input,output):
	g = rdflib.Graph()
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	[g.load(file(inp,"r"),format=_find_format(inp)) for inp in input]
	expanded_results = g.query(DIRECT_CLASSES)
	expanded_graph = rdflib.ConjunctiveGraph()
	for binding in expanded_results.bindings:
		discussion = rdflib.BNode()
		expanded_graph.add((discussion,TM['word'],binding['class']))
		expanded_graph.add((discussion,TM['date'],binding['date']))
		expanded_graph.add((discussion,TM['count'],binding['aggregated']))
	return expanded_graph

ALL_CLASSES = """
	PREFIX owl: <http://www.w3.org/2002/07/owl#>
	prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
	SELECT ?date ?class (SUM(?count) AS ?aggregated)
	WHERE {
		?instance tm:word ?word .
		?instance tm:date ?date .
		?instance tm:count ?count.
		?word owl:Class/rdfs:subClassOf* ?class
	} 
	GROUP BY ?class
"""

def all_classes(input,output):
	g = rdflib.Graph()
	TM = rdflib.Namespace("http://trandminer.org/ontology#")
	[g.load(file(inp,"r"),format=_find_format(inp)) for inp in input]
	expanded_results = g.query(ALL_CLASSES)
	expanded_graph = rdflib.ConjunctiveGraph()
	for binding in expanded_results.bindings:
		if not "class" in binding:
			embed()
			asdadssd
		discussion = rdflib.BNode()
		expanded_graph.add((discussion,TM['word'],binding['class']))
		expanded_graph.add((discussion,TM['date'],binding['date']))
		expanded_graph.add((discussion,TM['count'],binding['aggregated']))
	return expanded_graph