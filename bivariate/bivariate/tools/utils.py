import scipy.sparse as ssp
from IPython import embed
def reshapeflat(mat, shp):
	M,N = shp
	outv = ssp.lil_matrix(shp)
	for m in range(M):
		outv[m,:] = mat[0,m*N:(m+1)*N]
	return outv

def reshapecoo(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = ssp.coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b