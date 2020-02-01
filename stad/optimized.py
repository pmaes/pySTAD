import numpy as np
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.linalg.blas import dgemm


def unit_mst(distances):
    mst = minimum_spanning_tree(distances).toarray()
    mst = np.where(mst > 0, 1, 0)
    mst += mst.T
    mst = mst.astype('float32')
    return mst


def degrees(adj):
    return np.repeat(np.sum(adj, axis=1)[None, :], adj.shape[0], axis=0)


def triu_mask(m, k=0):
    mask = np.zeros_like(m, dtype=np.bool)
    idx = np.triu_indices_from(m, k=k)
    mask[idx] = True
    return mask


def masked_edges(m, mask):
    return np.logical_and(triu_mask(m, k=1), mask)


def ordered_edges(distances, mask):
    indices = np.nonzero(mask)
    ds = distances[indices]
    order = np.argsort(ds)
    return np.stack(indices, 1)[order].astype('int32')


def py_with_edges(adj_m, edges):
    new_adj = adj_m.copy()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        new_adj[x][y] = 1
        new_adj[y][x] = 1
    return new_adj


def seidel(adj):
    n = adj.shape[0]

    # Prepare our base case to compare against, an n x n matrix that is 0 on the
    # diagonal and 1 everywhere else.
    base_case = np.ones_like(adj)
    np.fill_diagonal(base_case, 0)
    
    # Let Z = A . A
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
    d = np.where(x >= np.multiply(t, degrees(adj)), 2 * t, 2 * t - 1)
    return d.astype('float32')


def sparsity(a):
    return 1.0 - (np.count_nonzero(a) / float(a.size))


def all_pair_shortest_paths(adj, method='auto', sparsity_threshold=0.9):
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


class STADObjective:
    def __init__(self, distances, mst, edges):
        self.distances = distances
        self.mst = mst
        self.edges = edges

        self.triu_indices = np.triu_indices(distances.shape[0], k=1)
        self.distance_vector = self.distances[self.triu_indices]

    def __call__(self, x):
        n = int(x)
        adj = py_with_edges(self.mst, self.edges[:n])
        dist = all_pair_shortest_paths(adj)
        dist = dist[self.triu_indices]
        return np.corrcoef(dist, self.distance_vector)[0][1]


# See:
# Malherbe, C., & Vayatis, N. (2017). Global optimization of Lipschitz functions. 34th International Conference on Machine Learning, ICML 2017, 5(1972), 3592–3601.
# https://github.com/UBC-CS/lipo-python/blob/master/src/sequential.py

def adaptive_lipo(func, bounds, n, p=0.1):
    # dimension of the domain
    d = len(bounds)

    alpha = 0.01/d
    k_seq=(1+alpha)**np.arange(-10000,10000) # Page 16

    # preallocate the output arrays
    y = np.zeros(n) - np.Inf
    x = np.zeros((n, d))
    loss = np.zeros(n)
    k_arr = np.zeros(n)
    
    # the lower/upper bounds on each dimension
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    # initialization with randomly drawn point in domain and k = 0
    k = 0
    k_est = -np.inf
    u = np.random.rand(d)
    x_prop = u * (bound_maxs - bound_mins) + bound_mins
    x[0] = x_prop
    y[0] = func(x_prop)
    k_arr[0] = k

    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x,axis=1))

    for t in np.arange(1, n):
        # draw a uniformly distributed random variable
        u = np.random.rand(d)
        x_prop = u * (bound_maxs - bound_mins) + bound_mins

        # check if we are exploring or exploiting
        if np.random.rand() > p: # enter to exploit w/ prob (1-p)
            # exploiting - ensure we're drawing from potential maximizers
            while upper_bound(x_prop, y[:t], x[:t], k) < np.max(y[:t]):
                u = np.random.rand(d)
                x_prop = u * (bound_maxs - bound_mins) + bound_mins 
        else:
            pass 
            # we keep the randomly drawn point as our next iterate
            # this is "exploration"
        # add proposal to array of visited points
        x[t] = x_prop
        y[t] = func(x_prop)
        loss[t] = np.max(y)

        # compute current number of tracked distances
        old_num_dist = (t * (t - 1)) // 2
        new_num_dist = old_num_dist + t

        new_x_dist = np.sqrt(np.sum((x[:t] - x[t])**2, axis=1))

        new_y_dist = np.abs(y[:t] - y[t])

        k_est = max(k_est, np.max(new_y_dist/new_x_dist))  # faster
        k = k_seq[np.argmax(k_seq >= k_est)]
        i_t = np.ceil(np.log(k_est)/np.log(1+alpha))
        k = (1+alpha)**i_t
        k_arr[t] = k
        

    output = {'loss': loss, 'x': x, 'y': y, 'k': k_arr}
    return output


def find_bounds(f, n, iterations, frac):
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


def stad(distances, unit=False, debug=False):
    if debug: print("Calculating MST")
    mst = unit_mst(distances)
    if debug: print("Removing MST links and sorting")
    without_mst = masked_edges(distances, mst == 0)
    edges = ordered_edges(distances, without_mst)
    objective = STADObjective(distances, mst, edges)

    if debug: print("Optimizing")
    #a, b = find_bounds(objective, edges.shape[0], 10, 0.90)
    a, b = 0, edges.shape[0]
    res = adaptive_lipo(objective, [(a, b)], 15, 0.1)

    opt_index = np.argmax(res['y'])
    opt_obj = res['y'][opt_index]
    opt = int(res['x'][opt_index])

    stad_adj = py_with_edges(mst, edges[:opt])
    if not unit:
        stad_adj = np.multiply(stad_adj, distances)

    return {
        'adj': stad_adj,
        'opt': {
            'x': opt,
            'obj': opt_obj},
        'obj': objective,
        'edges': edges,
        'x': res['x'],
        'y': res['y']
    }

def run_stad(highD_dist_matrix, lens=[], features={}, debug=False):
    import igraph

    res = stad(highD_dist_matrix, debug=debug)
    if debug: print(res['opt'])
    g = igraph.Graph.Weighted_Adjacency(res['adj'].tolist())
    
    distances = []
    for e in g.es():
        distances.append(highD_dist_matrix[e.source][e.target])
    g.es()['distance'] = distances

    return g