import bivariate.dataset.billdata as billdata
import scipy.sparse

def test_list_gather():
	nusers = 3
	ndays = 20
	nwords = 10
	userdayword = scipy.sparse.rand(
		nwords,nusers*ndays,density=0.1,format="csc"
	)

	suwd = billdata.suserdayword(
		userdayword,
		scipy.sparse.csc_matrix(userdayword.transpose()),
		ndays
	)


	l = suwd.list()
	assert len(l) == ndays