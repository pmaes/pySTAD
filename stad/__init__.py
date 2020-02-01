import pandas as pd
import igraph as ig
import numpy as np
from copy import deepcopy
from scipy import optimize


########
#### Create MST
########
def matrix_to_topright_array(matrix):
    for i, vector in enumerate(matrix):
        for j, value in enumerate(vector):
            if ( j > i ):
                yield value

def create_mst(dist_matrix, lens = [], features = {}):
    complete_graph = ig.Graph.Full(len(dist_matrix[0]))
    if len(lens) > 0:
        complete_graph.vs["lens"] = lens
    if not features == {}:
        for f in list(features.keys()):
            complete_graph.vs[f] = features[f]
    complete_graph.es["distance"] = list(matrix_to_topright_array(dist_matrix))
    return complete_graph.spanning_tree(weights = list(matrix_to_topright_array(dist_matrix)))

########
#### Alter distance matrix to incorporate lens
########
# There are 2 options to incorporate a lens:
# 1. Set all values in the distance matrix of datapoints that are in non-adjacent
#    bins to 1000, but leave distances within a bin and between adjacent
#    bins untouched.
# 2. a. Set all values in the distance matrix of datapoints that are in non-adjacent
#       bins to 1000, add 2 to the values of datapoints in adjacent bins, and
#       leave distances of points in a bin untouched.
#    b. Build the MST
#    c. Run community detection
#    d. In the distance matrix: add a 2 to some of the data-pairs, i.e. to those
#       that are in different communities.
#    e. Run the regular MST again on this new distance matrix (so is the same
#       as in step a, but some of the points _within_ a bin are also + 2)
def assign_bins(lens, nr_bins):
    # np.linspace calculates bin boundaries
    # e.g. bins = np.linspace(0, 1, 10)
    #  => array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
    #            0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
    bins = np.linspace(min(lens), max(lens), nr_bins)

    # np.digitize identifies the correct bin for each datapoint
    # e.g.
    #  => array([3, 9, 7, 8, 8, 3, 9, 6, 4, 3])
    return np.digitize(lens, bins)

def create_lensed_distmatrix_1step(matrix, assigned_bins):
    '''
    This will set all distances in non-adjacent bins to 1000. Data was
    normalised between 0 and 1, so 1000 is far (using infinity gives issues in
    later computations).
    Everything after this (building the MST, getting the list of links, etc)
    will be based on this new distance matrix.
    '''
    size = len(matrix)
    single_step_addition_matrix = np.full((size,size), 1000)

    for i in range(0, size):
        for j in range(i+1,size):
            if ( abs(assigned_bins[i] - assigned_bins[j]) <= 1 ):
                single_step_addition_matrix[i][j] = 0
    return matrix + single_step_addition_matrix

########
#### Evaluate result
########
def graph_to_distancematrix(graph):
    return graph.shortest_paths_dijkstra()

## Calculate correlation
def correlation_between_distance_matrices(matrix1, matrix2):
    '''
    correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(mst)))}
    '''
    return np.corrcoef(matrix1.flatten(), np.asarray(matrix2).flatten())[0][1]

def create_list_of_all_links_with_values(highD_dist_matrix):
    '''
    This function creates the full list:
    [
        { from: 0,
          to: 1,
          highDd: 1.0 },
        { from: 0,
          to: 2,
          highDd: 2.236 },
        ...
    '''
    all_links = []
    l = len(highD_dist_matrix[0])
    for i in range(0,l):
        for j in range(i+1, l):
            all_links.append({
                'from': i,
                'to': j,
                'highDd': highD_dist_matrix[i][j]
            })
    return all_links

## Remove links that are already in MST
def create_list_of_links_to_add(list_of_links, graph):
    '''
    This (1) removes all MST links and
         (2) sorts all based on distance.
    IMPORTANT!!
    Possible links are sorted according to their distance in the original
    high-dimensional space, and _NOT_ based on the error in the distances
    between the high-dimensional space and the MST.
    Example: below is the dataset from data/sim.csv, with the MST indicated.
    If we sort by the _error_, then the first link that will be added is
    between the points indicated with an 'o' (because they lie at the opposite
    ends of the MST). If we sort by distance in the original high-D space, the
    first link that will be added is between the points that are indicated
    with a 'v'.

                *--*--*--*
                |        |
          *--o  *        *
          |     |        |
          *     *        *
          |     |        |
    *--*--*--v  v  o--*--*
       |        |
       *        *
       |        |
       *--*--*--*

    '''

    output = deepcopy(list_of_links)
    ## Remove the links that are already in the MST
    for e in graph.es():
        elements = list(filter(lambda x:x['to'] == e.target and x['from'] == e.source, list_of_links))
        output.remove(elements[0])
    ## Sort the links based on distance in original space
    output.sort(key = lambda x: x['highDd']) ## IMPORTANT!
    return output

## Add links to graph
def add_links_to_graph(graph, highD_dist_matrix, list_of_links_to_add, n):
    new_graph = deepcopy(graph)
    new_graph.add_edges(list(map(lambda x:(x['from'],x['to']), list_of_links_to_add[:int(n)])))
    distances = []
    for e in new_graph.es():
        distances.append(highD_dist_matrix[e.source][e.target])
    new_graph.es()['distance'] = distances
    return new_graph

## Using basinhopping
def cost_function(nr_of_links, args):
    graph = args['graph']
    list_of_links_to_add = args['list_of_links_to_add']
    highD_dist_matrix = args['highD_dist_matrix']

    new_graph = add_links_to_graph(graph, highD_dist_matrix, list_of_links_to_add, nr_of_links)
    return 1 - correlation_between_distance_matrices(highD_dist_matrix, list(graph_to_distancematrix(new_graph)))

def run_basinhopping(cf, mst, links_to_add, highD_dist_matrix, debug = False):
    '''
    Returns new graph.
        cf = cost_function
        start = start x
    '''
    disp = False
    if debug: disp = True
    start = len(mst.es())
    minimizer_kwargs = {'args':{'graph':mst,'list_of_links_to_add':links_to_add,'highD_dist_matrix':highD_dist_matrix}}
    result = optimize.basinhopping(
        cf,
        start,
        disp=disp,
        minimizer_kwargs=minimizer_kwargs
    )
    if debug:
        print(result)
    g = add_links_to_graph(mst, highD_dist_matrix, links_to_add, result.x[0])
    return g

########
#### Bringing everything together
########
def run_stad(highD_dist_matrix, lens=[], features={}, debug=False):
    '''
    Options:
    * `lens` needs to be an array with a single numerical value for each datapoint,
       or an empty array
    * `features` is a list of feature that will be added to the
       node tooltip. Format: {'label1': [value1, value2], 'label2': [value1, value2]}
    '''
    ## Check if distance matrix is normalised
    if ( np.min(highD_dist_matrix) < 0 or np.max(highD_dist_matrix) > 1 ):
        print("ERROR: input distance matrix needs to be normalised between 0 and 1")
        exit()

    if debug: print("Tweaking distance matrix if we're working with a lens")
    if len(lens) > 0:
        assigned_bins = assign_bins(lens, 5)
        dist_matrix = create_lensed_distmatrix_1step(highD_dist_matrix, assigned_bins)
    else:
        dist_matrix = highD_dist_matrix
    if debug: print("Calculating MST")
    mst = create_mst(dist_matrix, lens, features)
    if debug: print("Creating list of all links")
    all_links = create_list_of_all_links_with_values(dist_matrix)
    if debug: print("Removing MST links and sorting")
    list_of_links_to_add = create_list_of_links_to_add(all_links, mst)
    if debug: print("Start basinhopping")
    g = run_basinhopping(
            cost_function,
            mst,
            list_of_links_to_add,
            dist_matrix,
            debug=debug)
    return g