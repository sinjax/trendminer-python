from model import *
import argparse
from expand_triples import direct_classes, all_classes
import os
import logging;logger = logging.getLogger("root")
import shutil
from rdfutil import construct_word_graph

parser = argparse.ArgumentParser(description='Expand annotations')

subparsers = parser.add_subparsers()

parser_direct = subparsers.add_parser('direct')
parser_direct.set_defaults(expand=direct_classes)
parser_all = subparsers.add_parser('all')
parser_all.set_defaults(expand=all_classes)

def addinout(parser):
	parser.add_argument('-out', dest="out",type=str,default="out")
	parser.add_argument('-fmt', dest="rdftype",type=str,default="turtle")
	parser.add_argument('-in', dest="inp",type=str,required=True,nargs="+")
	parser.add_argument('-ext', dest="extras",type=str,nargs="+")
	parser.add_argument('-cpy', dest="to_copy",default=[],type=str,nargs="*")

addinout(parser_direct)
addinout(parser_all)

options = parser.parse_args()

if not os.path.exists(options.out): os.makedirs(options.out)

words = {}

for inp in options.inp:
	graph_name = os.path.basename(inp)
	logger.debug("Expanding: " + graph_name)
	expanded = options.expand(
		[inp] + options.extras,options.out,
		words=words
	)
	expanded.serialize(
		destination=os.sep.join(
			[
				options.out,
				"%s.%s"%(
					graph_name.split(".")[0],
					options.rdftype
				)
			]
		),
		format=options.rdftype
	)
logger.debug("Creating word graph...")
words = construct_word_graph(words)
words.serialize(
	destination=os.sep.join(
		[
			options.out,
			"word.%s"%(
				options.rdftype
			)
		]
	),
	format=options.rdftype
)

for cpy in options.to_copy: 
	logger.debug("Copying: %s"%cpy)
	shutil.copy(cpy,options.out)