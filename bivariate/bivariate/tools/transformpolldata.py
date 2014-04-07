#!/usr/bin/env python
from bivariate.dataset import userwordregion
from bivariate.dataset import regionpolls as rpt
from bivariate.dataset import genderpolls as gpt
from scipy import io as sio
from scipy import sparse as ssp
from IPython import embed
import os
import logging;logger = logging.getLogger("root")
POLL_DATA = "/home/ss/Experiments/bilinear/polls_demographics.mat"
from optparse import OptionParser



parser = OptionParser()
parser.add_option("-p", "--poll-data", dest="poll_file", default=POLL_DATA, metavar="FILE",
                  help="A matlab file containing region polls as matrices with 'region' in their name")
parser.add_option("-n", "--number-of-days", dest="ndays", default=15,
                  help="Number of days")
parser.add_option("-o", "--output", dest="output", default=".",
                  help="Root directory to output the matrices")

(options, args) = parser.parse_args()

if not os.path.exists(options.poll_file):
	raise Exception("File's not found!")


logger.debug("Loading polls")
regionpolls = rpt.transform(options.poll_file)
genderpolls = gpt.transform(options.poll_file)
if not os.path.exists(options.output): os.makedirs(options.output)
Y_out = "%s/Y.mat"%options.output

logger.debug("Outputting poll information to: %s"%Y_out)
sio.savemat(
	Y_out,
	{
		"regionpolls":regionpolls,
		"genderpolls":genderpolls
	}
)
