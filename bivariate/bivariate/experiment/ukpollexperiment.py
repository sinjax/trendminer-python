from bivariate.tools.utils import *
from ..dataset.userwordregion import *
from IPython import embed
import logging;logger = logging.getLogger("root")

Xfile = "/data/ss/UK_regions_Sina/prepared/X.mat"
Yfile = "/data/ss/UK_regions_Sina/prepared/Y.mat"

logger.debug("Loading X data...")
Xmats = h5py_sparse_row(Xfile)



def prepareFolds(ndays, nfolds):
	set_fold = [];
	step = 5;               // % test_size
	t_size = 48;            // % training_size
	v_size = 8;
	for i in range(nfolds):
		total = i * step + t_size;
		training = range(total - v_size)
		test = new range(step);
		validation = range(v_size);
		j = 0;	
		traini = 0;
		tt = round(total/2.)-1;
		while j < tt - vsize/2:
			training[traini] = j;
			j+=1
			traini+=1

		while k < len(validation):
			validation[k] = j;
			k+=1
			j+=1

		while j < total:
			training[traini] = j;
			j+=1
			traini+=1

		while k < len(test):
			test[k] = j;
			k+=1
			j+=1
		foldi = {
			"training":training
			"test": test
			"validation":validation
		}
		set_fold += [foldi]
		return set_fold;
	}