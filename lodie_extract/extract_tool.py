
import argparse
import extract_single
parser = argparse.ArgumentParser(description='Extract annotations from files')
parser.add_argument('files', nargs='+')

options = parser.parse_args()

for f in options.files:
	extract_single.extract_single(f)