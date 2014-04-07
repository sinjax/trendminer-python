from ..dataset.userwordregion import read_split_userwordregion
from ..dataset.userwordgender import read_split_userwordgender
def read_regionpoll(pollmat):
	return pollmat['regionpolls']

def read_genderpoll(pollmat):
	return pollmat['genderpolls']
def prepare_input_mode(parser):
	subparsers = parser.add_subparsers()
	
	parser_region = subparsers.add_parser('region')
	parser_region.set_defaults(readsplit=read_split_userwordregion,readpoll=read_regionpoll)
	parser_gender = subparsers.add_parser('gender')
	parser_gender.set_defaults(readsplit=read_split_userwordgender,readpoll=read_genderpoll)