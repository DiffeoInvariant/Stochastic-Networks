import scipy.io as sp
from scipy.sparse.linalg import eigs
from scipy.special import expit
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read_graph(filename, return_adj_mat=True):
    # takes .mtx, .mtx.gz files
    adjmat = sp.mmread(filename)

    adjmat = adjmat.ceil()

    G = nx.from_scipy_sparse_matrix(adjmat, edge_attribute='distance')
    if return_adj_mat:
        return G, adjmat.tocsr()
    else:
        return G


def get_coords(filename, coord_map_function=None):
    if coord_map_function is None:
        return 1.0e4 * sp.mmread(filename)
    else:
        return coord_map_function(sp.mmread(filename))

def add_coords_to_graph(g, filename, coord_map_function=None):
    
    coords = get_coords(filename, coord_map_function)
    g.position = {}

    count = 0
    for v in list(g.nodes):
        g.position[v] = (coords[count,0], coords[count,1])
        count += 1
    return g

def plot_networkx_graph(g, figure_size=(8,8)):
    plt.figure(figsize=figure_size)
    #color based on vertex degree
    node_color = [float(g.degree(v)) for v in g]
    node_size = [float(g.degree(v)) for v in g]

   # spr_pos = nx.spring_layout(g, pos=g.position)
    nx.draw(g, g.position, node_color=node_color, node_size=node_size)

    plt.show()

def get_node_mapping(filename, special_index=None):
    #special_index is a list of strings
    mapping = {}
    special_id = []
    with open(filename) as f:
        for num, line in enumerate(f):
            if special_index is not None:
                for i in special_index:
                    if i in str(line):
                        special_id.append(num)
            mapping[num] = str(line)

    return mapping if special_index is None else mapping, special_id

def name_nodes(g, filename, return_names=False, special_index=None):

    if special_index is None:
        mapping = get_node_mapping(filename)
    else:
        mapping, special_id = get_node_mapping(filename, special_index)

    if return_names:
        if special_index is not None:
            return nx.relabel_nodes(g, mapping, copy=False), [str(v).rstrip() for v in mapping.values()], special_id
        else:
            return nx.relabel_nodes(g, mapping, copy=False), [str(v).rstrip() for v in mapping.values()]
    else:
        if special_index is not None:
            return nx.relabel_nodes(g, mapping, copy=False), special_id
        else:
            return nx.relabel_nodes(g, mapping, copy=False)


def get_k_largest_indices(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    #sort
    return indices[np.argsort(arr[indices])]


if __name__ == '__main__':

    g, adjmat = read_graph('../netsim/data/USAir97/USAir97.mtx')

    rows, cols = adjmat.shape

    assert(rows == cols)

    print("Graph has %d nodes." % rows)

    g, names, airport_ids = name_nodes(g, '../netsim/data/USAir97/USAir97_nodename.txt', True, ['Stapleton', 'Kwigillingok'])

    #S comes after K, so Stapleton will be second
    stapleton_id = airport_ids[1]
    
    kwig_id = airport_ids[0]
    
    def coord_mapper(mat):
        for i in range(rows):
            #for j in range(2):
             mat[i,1] = 1- mat[i,1]
        return mat


    g = add_coords_to_graph(g, '../netsim/data/USAir97/USAir97_coord.mtx', coord_mapper)

    #get all eigenvalues since adjmat is small (4-5 thousand nonzeros)
    evals, evecs = eigs(adjmat.toarray(), k=rows)
    print("The adjacency matrix has %d eigenvalues.\n" % evals.shape)
    print("Largest eigenvalue: %a." % evals[0])
    #eigenvector for centrality purposes
    cent_eig_vec = evecs[:,0]

    #most important 5 indices by evec centrality
    important_indices = get_k_largest_indices(cent_eig_vec, 5)
    print("\nMost important airports by eigenvector centrality:")
    count = 1
    for i in important_indices:
        print()
        print(count, names[i])
        count += 1

    #degree centrality
    cent_dict = nx.degree_centrality(g)
    sorted_dict = collections.OrderedDict(sorted(cent_dict.items(), key=lambda x: x[1], reverse=True))
    print("\nMost important airports by degree centrality:\n")
    keys = list(sorted_dict.keys())
    for i in range(5):
        print(i+1, keys[i])

    one = np.ones(cols)

        
    DEN_degree = adjmat[stapleton_id,:].count_nonzero()#int(sum(adjmat[stapleton_id,:] * one[:]))

    print("Number of airports one flight away from Denver: ", DEN_degree)

    #very small matrix, so really who cares about efficiency
    A2 = adjmat @ adjmat

    #can't get to more airports than exist...
    DEN_len2 = A2[stapleton_id,:].count_nonzero()  

    print("Number of airports two flights away from Denver: ", DEN_len2)

    A3 = A2 @ adjmat

    DEN_len3 = A3[stapleton_id,:].count_nonzero()  

    print("Number of airports three flights away from Denver: ", DEN_len3)

    A4 = A3 @ adjmat

    DEN_len4 = A4[stapleton_id,:].count_nonzero() 
    
    print("Number of airports three flights away from Denver: ", DEN_len4)

    kwig = list(g.nodes)[kwig_id]

    kwig_max_path_length = max(nx.shortest_path_length(g, source=kwig).values())

    print("Maximum path length from Kwillignok: ", kwig_max_path_length)

    num_triangles = np.trace(A3.toarray()) / 6.0 #divide by 6 cuz 3 places to start each triangle, 2 directions to go in

    num_triplets = 0.5*sum(one.T * A2 * one)

    print("Number of (triangles, triplets): ", (num_triangles, num_triplets))

    clust_coeff = 3* num_triangles / num_triplets

    print("Clustering coefficient: ", clust_coeff)
        
    cent_deg_vals = [np.absolute(i) for i in list(cent_dict.values())]
    
    plt.figure(figsize=(8,8))
    plt.scatter(cent_deg_vals, cent_eig_vec)
    plt.ylabel("Eigenvector centrality")
    plt.xlabel("Degree centrality")


    plt.figure(figsize=(8,8))
    degrees = [g.degree(n) for n in g.nodes]
    plt.hist(degrees, bins=40)
    plt.title("Degree distribution")
    
    plot_networkx_graph(g, figure_size=(12,7))
