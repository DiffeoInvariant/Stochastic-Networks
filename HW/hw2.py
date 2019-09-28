import networkx as nx
import numpy as np
from scipy.optimize import fsolve as solve
import matplotlib.pyplot as plt

def get_max_cc_size(G):
    return len(list((max(nx.connected_component_subgraphs(G), key=len).nodes).values()))

if __name__ == '__main__':
    
    print('Problem 3a')
    fbgr = nx.read_edgelist('facebook_data.txt', create_using=nx.DiGraph(), delimiter=',')
    kout = np.asarray([i[1] for i in list(fbgr.out_degree())])
    fbam = np.asarray(nx.adjacency_matrix(fbgr).todense())
    khat = (np.matmul(kout.T, fbam))/kout
    plt.figure()
    plt.scatter(kout, khat)
    plt.xlabel('kout')
    plt.ylabel('khat')

    print('Problem 3b')
    r = nx.degree_pearson_correlation_coefficient(fbgr)
    print('Assortativity coefficient: ', r)

    print('Problem 3c')
    kin = np.asarray([i[1] for i in list(fbgr.in_degree())])

    plt.figure()
    plt.scatter(kout, kin)
    plt.xlabel('kout')
    plt.ylabel('kin')
    
    print('Problem 4a')
    n = 500
    k = 3.0
    p = k/float(n)

    g = nx.fast_gnp_random_graph(n, p)

    gccsize = get_max_cc_size(g)

    print("Largest cc size divided by n: ", float(gccsize)/float(n))

    print('Problem 4b')

    krange = [0.1 + 0.01 * 2.4 * i for i in range(101)]

    lcnodefrac = []
    sfuncsols = []

    def get_sfunc(k):
        return lambda x: 1 - np.exp(-k * x) - x

    n = 10000
    for kval in krange:
        p = kval/float(n)
        g = nx.fast_gnp_random_graph(int(n), p)
        lcnodefrac.append(get_max_cc_size(g)/float(n))
        sf = get_sfunc(kval)
        sfuncsols.append(solve(sf, 0.5*k))

    plt.figure()
    plt.scatter(krange, lcnodefrac)
    plt.plot(krange, sfuncsols, color='red')
    plt.xlabel('k')
    plt.ylabel('fraction of nodes in largest connected component')
    
    
    plt.show()
