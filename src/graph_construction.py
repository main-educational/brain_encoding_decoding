import numpy as np
import torch
import torch_geometric as tg


def _make_undirected(mat):
    """
    Takes an input adjacency matrix and makes it undirected (symmetric).

    Parameter
    ----------
    mat: array
        Square adjacency matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    sym = (mat + mat.transpose()) / 2
    if len(np.unique(mat)) == 2:  # if graph was unweighted, return unweighted
        return np.ceil(sym)  # otherwise return average
    return sym


def _knn_graph_quantile(mat, self_loops=False, k=8, symmetric=True):
    """
    Takes an input correlation matrix and returns a k-Nearest
    Neighbour weighted undirected adjacency matrix.
    """

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    dim = mat.shape[0]
    if (k <= 0) or (dim <= k):
        raise ValueError("k must be in range [1,n_nodes)")
    is_directed = not (mat == mat.transpose()).all()
    if is_directed:
        raise ValueError(
            "Input adjacency matrix must be undirected (matrix symmetric)!"
        )

    # absolute correlation
    mat = np.abs(mat)
    adj = np.copy(mat)
    # get NN thresholds from quantile
    quantile_h = np.quantile(mat, (dim - k - 1) / dim, axis=0)
    mask_not_neighbours = mat < quantile_h[:, np.newaxis]
    adj[mask_not_neighbours] = 0
    if not self_loops:
        np.fill_diagonal(adj, 0)
    if symmetric:
        adj = _make_undirected(adj)
    return adj


def make_group_graph(connectomes, k=8, self_loops=False, symmetric=True):
    """
    Parameters
    ----------
    connectomes: list of array
        List of connectomes in n_roi x n_roi format, connectomes must all be the same shape.
    k: int, default=8
        Number of neighbours.
    self_loops: bool, default=False
        Wether or not to keep self loops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.

    Returns
    -------
    Torch geometric graph object of k-Nearest Neighbours graph for the group average connectome.
    """
    if connectomes[0].shape[0] != connectomes[0].shape[1]:
        raise ValueError("Connectomes must be square.")

    # Group average connectome and nndirected 8 k-NN graph
    avg_conn = np.array(connectomes).mean(axis=0)
    avg_conn = np.round(avg_conn, 6)
    avg_conn_k = _knn_graph_quantile(
        avg_conn, k=k, self_loops=self_loops, symmetric=symmetric
    )

    # Format matrix into graph for torch_geometric
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(avg_conn_k))
    return tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])
