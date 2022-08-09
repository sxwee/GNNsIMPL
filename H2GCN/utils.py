from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
import scipy.sparse as sp


def toCSR(spt):
    if not spt.has_value():
        spt = spt.fill_value(1.)
    rowptr, col, value = spt.csr()
    mat = sp.csr_matrix((value, col, rowptr)).tolil()
    # remove self-loops
    mat.setdiag(0)
    return mat.tocsr()


def hopNeighborhood(adj):
    adj2 = adj.matmul(adj).fill_value(1.0)
    adj_2hop = (toCSR(adj2) - toCSR(adj)) > 0
    adj_2hop = SparseTensor.from_scipy(adj_2hop).fill_value(1.0)
    return adj2


def norm_adj(adj_t, add_self_loops=True):
    """
    normalization adj
    """
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


if __name__ == "__main__":
    pass
