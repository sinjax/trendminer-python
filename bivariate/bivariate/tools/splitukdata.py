from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")

Xfile = "/data/ss/UK_regions_Sina/prepared/X.mat"
outroot = "/data/ss/UK_regions_Sina/prepared/X_expand_scipy"

logger.debug("loading data")
Xmats = h5py_sparse_row(Xfile)

split_userwordregion(Xmats,235,outroot)