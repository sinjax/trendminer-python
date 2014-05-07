from model import *
import argparse
from expand_triples import direct_classes
import os

parser = argparse.ArgumentParser(description='Expand annotations')

subparsers = parser.add_subparsers()

parser_direct = subparsers.add_parser('direct')
parser_direct.set_defaults(expand=direct_classes)

def addinout(parser):
	parser.add_argument('-out', dest="out",type=str,default="out")
	parser.add_argument('-fmt', dest="rdftype",type=str,default="turtle")
	parser.add_argument('-in', dest="inp",type=str,required=True,nargs="+")

addinout(parser_direct)

options = parser.parse_args()
expanded = direct_classes(options.inp,options.out)
if not os.path.exists(options.out): os.makedirs(options.out)
graph_name = os.path.basename(options.inp[0])
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