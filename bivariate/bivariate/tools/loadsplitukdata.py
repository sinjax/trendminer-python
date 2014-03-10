from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")
import sys

inputroot = "/data/ss/UK_regions_Sina/prepared/X_expand_scipy"
if len(sys.argv) > 1:
	inputroot = sys.argv[1]

days = range(0,48)
dataCache = {}
logger.debug("Reading First time...")
X = read_split_userwordregion(inputroot,cache=dataCache,*days)
logger.debug("Reading again...")
X = read_split_userwordregion(inputroot,cache=dataCache,*days)
