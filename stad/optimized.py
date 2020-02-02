import numpy as np
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.linalg.blas import dgemm

from stad import create_lensed_distmatrix_1step, assign_bins
import stad.lipo as lipo
import stad.diff_curve as diff_curve

# This module aims to implement an optimized version of the base STAD algorithm.
# There are three core ideas to achieve this:
# 1. Operate directly on the adjacency matrix.
#    All functions operate directly on the numpy adjacency matrix, saving us the
#    roundtrip through python data structures.
#
# 2. Compute the All Pairs Shortest Paths faster.
#    This the main bottleneck in computing the fitness of a given amount of
#    edges. By implementing the Seidel algorithm, we get a big speed increase on
#    dense graphs.

# 3. Use an optimizing algorithm that requires less function evaluations.
#    Simulated annealing and basin hopping assume fast objective functions and
#    evaluate these liberally. 
#
#    LIPO is a global optimization algorithm that is efficient in function calls
#    and requires little parameters.
#
#    Diff_curve is designed for this specific objective function (broadly
#    concave, increasing) and has a bias towards a local maximum at a lower
#    amount of edges.


# Linear algebra libraries like BLAS tend to be optimized for float32.
# We cast distance and adjacency matrices to make sure they are of this type.

def unit_mst(distances):
    """For a given distance matrix, returns the unit-distance MST."""
    # Use toarray because we want a regular nxn matrix, not the scipy sparse matrix.
    mst = minimum_spanning_tree(distances).toarray()
    # Set every edge weight to 1.
    mst = np.where(mst > 0, 1, 0)
    # Symmetrize.
    mst += mst.T
    mst = mst.astype('float32')
    return mst

# These masking functions are utilities to ease the removal of the MST edges.

def triu_mask(m, k=0):
    """
    For a given matrix m, returns like-sized boolean mask that is true for all
    elements `k` offset from the diagonal.
    """
    mask = np.zeros_like(m, dtype=np.bool)
    idx = np.triu_indices_from(m, k=k)
    mask[idx] = True
    return mask


def masked_edges(adj, mask):
    """
    For a given adjacency matrix and a like-sized boolean mask, returns a mask
    of the same size that is only true for the unqiue edges (the upper
    triangle). Assumes a symmetrical adjacency matrix.
    """
    return np.logical_and(triu_mask(adj, k=1), mask)


def ordered_edges(distances, mask):
    """
    For a given adjacency matrix with `n` edges and a like-sized mask, returns
    an n x 2 array of edges sorted by distance.
    """
    # We are only interested in the indicies where our mask is truthy.
    # On a boolean array nonzero returns the true indices.
    # indices holds a tuple of arrays, one for each dimension.
    indices = np.nonzero(mask)
    ds = distances[indices]
    # argsort returns the sorted indices of the distances.
    # Note: these are not the same as the indices of our mask.
    order = np.argsort(ds)
    # We wish to return a single array, so we use `stack` to combine the two nx1
    # arrays into one nx2 array.
    combined_indices = np.stack(indices, 1)
    # Finally, we reorder our combined indices to be in the same order as the sorted distances.
    return combined_indices[order].astype('int32')


def with_edges(adj_m, edges):
    """
    For a given adjacency matrix and an nx2 array of edges, returns a new
    adjacency matrix with the edges added. Symmetrizes the edges.
    """
    new_adj = adj_m.copy()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        new_adj[x][y] = 1
        new_adj[y][x] = 1
    return new_adj


def seidel(adj):
    """
    Implements the Seidel algorithm for APSP on unweighted, undirected graphs.
    For a given adjacency matrix, returns a like-sized matrix holding the
    distances of the shortest paths between nodes.
    """
    n = adj.shape[0]

    # Prepare our base case to compare against, an n x n matrix that is 0 on the
    # diagonal and 1 everywhere else.
    base_case = np.ones_like(adj)
    np.fill_diagonal(base_case, 0)
    
    # Let Z = A . A
    # Using DGEMM directly here instead of np.dot because the latter is 10x slower.
    z = dgemm(1, adj, adj)

    # Let B be an n x n binary matrix
    # where b_ij =
    #   1 if it isn't on the diagonal and a_ij = 1 or z_ij > 0
    #   0 otherwise
    b = np.where(
        np.logical_and(
            # Not on the diagonal
            np.logical_not(np.eye(n)),
            np.logical_or(adj == 1, z > 0)),
        1,
        0)

    # If all b_ij not on the diagonal are 1
    if (b == base_case).all():
        return 2 * b - adj

    t = seidel(b)

    # Let X = T . A
    x = dgemm(1, t, adj)

    # Return n x n matrix D
    # where d_ij =
    #   2*t_ij     if x_ij >= t_ij * degree(j)
    #   2*t_ij - 1 if x_ij <  t_ij * degree(j)
    degrees = np.repeat(np.sum(adj, axis=1)[None, :], adj.shape[0], axis=0)
    d = np.where(x >= np.multiply(t, degrees), 2 * t, 2 * t - 1)
    return d.astype('float32')


# The Seidel algorithm is significantly faster than BFS/Dijkstra on dense
# graphs, but is slower on sparse ones. To pick between these somewhat
# intelligently, we calculate the sparsity of the adjacency matrix and pick
# Seidel if it's below a certain threshold.

def sparsity(a):
    return 1.0 - (np.count_nonzero(a) / float(a.size))

# A default of 0.9 seems to be close to the actual threshold on my system.
def all_pair_shortest_paths(adj, method='auto', sparsity_threshold=0.9):
    """Calculate the APSP for a given adjacency matrix."""
    if method == 'auto':
        s = sparsity(adj)
        # Naive
        if s < sparsity_threshold:
            method = 'seidel'
        else:
            method = 'dijkstra'

    if method == 'seidel':
        return seidel(adj)
    elif method == 'dijkstra':
        return dijkstra(adj).astype('float32')

# We frame the STAD optimization as finding the optimal number of edges to add.
# To evaluate the STAD objective function for a given number of edges we need
# the high-dimensional distances to compare against, the MST to construct the
# network from, and the sorted array of edges to add. This state is kept in the
# STADObjective method. Because it implements __call__ it can be used as a
# regular function. It might be helpful to think of it a as a closure over the
# aforementioned bits of data.


class STADObjective:
    """The STAD objective function to maximalize"""
    def __init__(self, distances, mst, edges):
        self.distances = distances
        self.mst = mst
        self.edges = edges

        # It's worth it to only compare the unique edges. The indexing isn't
        # cheap, but indexing and calculating the correlation is still cheaper
        # than calculating the correlation for the full matrix.
        self.triu_indices = np.triu_indices(distances.shape[0], k=1)
        # We can cache the unique distances here, since they are constant.
        self.distance_vector = self.distances[self.triu_indices]
    
    @property
    def max_n(self):
        return self.edges.shape[0]-1

    def __call__(self, x):
        # Force integer indices, so this objective can be used with general optimization methods.
        n = int(x)
        # Construct the adjacency matrix with the given number of edges.
        adj = with_edges(self.mst, self.edges[:n])
        # Calculate the distance matrix thereof.
        dist = all_pair_shortest_paths(adj)
        dist = dist[self.triu_indices]
        # Return the correlation between these distances and the high-dimensional ones.
        # This means the function should be maximalised.
        return np.corrcoef(dist, self.distance_vector)[0][1]


# Because we assume our objective function will never increase to a new maximum
# after a certain point, we can estimate a quick bound before running more
# expensive optimization.

# restrict_bounds evaluates the objective in steps determined by the amount of
# iterations we maximally wish to spend, and stops and returns once the next
# value is a certain fraction lower than the maximum found.
# It returns the region that is within `frac` of the maximum.
def restrict_bounds(f, n, iterations, frac):
    best = 0
    xs, ys = [], []
    
    for x in np.linspace(0, n-1, iterations, dtype=np.int):
        y = f(x)
        xs.append(x)
        ys.append(y)

        if y > best:
            best = y

        if y/best < frac:
            break

    xs = np.array(xs)
    ys = np.array(ys)

    a = xs[np.argmax(ys/np.max(ys) > frac)]
    b = xs[-1]
    return (a, b)


# We need a common API for our optimizers. These optimize_ functions wrap actual
# optimizing algorithms so that they only take the objective function and a
# debug flag. An obvious improvement would be to allow passing on of parameters,
# which are currently hardcoded.


def optimize_lipo(objective, debug=False):
    if debug: print("Restricting bounds")
    a, b = restrict_bounds(objective, objective.max_n, 10, 0.90)
    if debug: print("LIPO")
    res = lipo.adaptive_lipo(objective, [(a, b)], 15, 0.1)
    opt_index = np.argmax(res['y'])
    opt_obj = res['y'][opt_index]
    opt = int(res['x'][opt_index])

    return {'x': opt, 'y': opt_obj}


def optimize_diff_curve(objective, debug=False):
    return diff_curve.optimize_diff_curve(objective, objective.max_n, debug=debug)

# To allow for dynamic choice of optimizer, we map their names to the actual functions.

OPTIMIZERS = {
    'lipo': optimize_lipo,
    'diffcurve': optimize_diff_curve
}


def stad(distances, unit=False, debug=False, opts={}):
    if debug: print("Calculating MST")
    mst = unit_mst(distances)
    if debug: print("Removing MST links and sorting")
    without_mst = masked_edges(distances, mst == 0)
    edges = ordered_edges(distances, without_mst)
    objective = STADObjective(distances, mst, edges)

    optimizer = opts.get('optimizer', 'diffcurve')
    if debug: print(f"Optimizing using {optimizer}")
    optimizer_f = OPTIMIZERS[optimizer]
    opt = optimizer_f(objective, debug=debug)

    stad_adj = with_edges(mst, edges[:opt['x']])
    if not unit:
        stad_adj = np.multiply(stad_adj, distances)

    return {
        'adj': stad_adj,
        'opt': opt,
        'obj': objective,
        'edges': edges,
    }


# This wrapper exists to give a similar API to the base implementation.

def run_stad(highD_dist_matrix, lens=[], features={}, debug=False, opts={}):
    import igraph

    res = stad(highD_dist_matrix, debug=debug, opts=opts)
    if debug: print(res['opt'])
    g = igraph.Graph.Weighted_Adjacency(res['adj'].tolist())

    distances = []
    for e in g.es():
        distances.append(highD_dist_matrix[e.source][e.target])
    g.es()['distance'] = distances

    return g