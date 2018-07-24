import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import tensorflow as tf
import ast
import scipy
from numpy.linalg import svd, qr, norm
import glob
from collections import defaultdict
from math import log


def change(p, w, hnodes, nodes, bin_dim, degree, indicator):
    p_matrix = np.zeros((nodes - len(hnodes), nodes - len(hnodes)))
    w_matrix = np.zeros((nodes - len(hnodes), nodes - len(hnodes), bin_dim))
    
    indicator_new = np.zeros((nodes - len(hnodes), bin_dim))
    
    degree_new = np.zeros(nodes - len(hnodes))
    rest = list(set(range(nodes)) - set(hnodes)) 
    print("Debig dim", rest, len(rest), p_matrix.shape)
    k = 0
    for i in rest:
        l = 0
        degree_new[k] = degree[i]
        indicator_new[k] = indicator[i]
        for j in rest:
                p_matrix[k][l] = p[i][j]
                w_matrix[k][l] = w[i][j]
                #indicator_new[k][l] = indicator[i][j]
                l += 1
        k += 1 
    return (p_matrix, w_matrix, degree_new, indicator_new)


def normalise_h2(prob, weight, bin_dim, indicator, edge_mask, list_edges):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(prob, 10 * temp))

    
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    combined_problist = []
    problist = []
    candidate_list_edges = []
    for (u,v) in list_edges:
        for i in range(bin_dim):
            candidate_list_edges.append((u, v, i+1))
        problist.append(p_rs[u][v]*edge_mask[u][v])
        
        indi = np.multiply(indicator[u], indicator[v])
        denom = sum(np.multiply(w_rs[u][v], indi))
        if denom == 0:
            denom = 1
            #del problist[-1]
        w_rs[u][v] = np.multiply(w_rs[u][v], indi) / denom
        combined_problist.extend(p_rs[u][v] * edge_mask[u][v] * w_rs[u][v])

    combined_problist = np.array(combined_problist)
    return combined_problist / combined_problist.sum(), candidate_list_edges


def normalise_h1(prob, weight, bin_dim, indicator, edge_mask, node):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(prob, 10 * temp))
   
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    combined_problist = []
    problist = []

    for j in range(n):
        if j != node:
            if j < node:
                problist.append(p_rs[j][node] * edge_mask[j][node])
                indi = np.multiply(indicator[node], indicator[j])
                denom = sum(np.multiply(w_rs[j][node], indi))
                if denom == 0:
                    denom = 1
                    del problist[-1]
                w_rs[j][node] = np.multiply(w_rs[node][j], indi) / denom
                combined_problist.extend(p_rs[j][node] * w_rs[j][node] * edge_mask[j][node])
            else:
                problist.append(p_rs[node][j] * edge_mask[node][j])
                indi = np.multiply(indicator[node], indicator[j])
                denom = sum(np.multiply(w_rs[node][j], indi))
                if denom == 0:
                    denom = 1
                    del problist[-1]
                w_rs[node][j] = np.multiply(w_rs[node][j], indi) / denom
                combined_problist.extend(p_rs[node][j] * w_rs[node][j] * edge_mask[j][node])
    problist = np.array(problist)
    combined_problist = np.array(combined_problist)
    return combined_problist / combined_problist.sum()


def get_weighted_edges(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat):
        i = 0
        candidate_edges = []
        list_edges = get_candidate_edges(len(node_list))
        try:
            while i < n_edges :
                p = normalise_h1(prob, w_edge, indicator.shape[1], indicator, edge_mask)
                candidate_edges.extend([list_edges[k] for k in
                                    np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])

                (u, v, w) = candidate_edges[i]
                degree_mat[u] += w
                degree_mat[v] += w

                edge_mask[u][v] = 0
                edge_mask[v][u] = 0

                if (node_list[u] - degree_mat[u]) == 0:
                    indicator[u][0] = 0
                if (node_list[u] - degree[u]) <= 1:
                    indicator[u][1] = 0
                if (node_list[u] - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (node_list[v] - degree[v]) == 0:
                    indicator[v][0] = 0
                if (node_list[v] - degree[v]) <= 1:
                    indicator[v][1] = 0
                if (node_list[v] - degree[v]) <= 2:
                    indicator[v][2] = 0
                i += 1
        except:
            # if the alignment was no possible
            candidate_edges = []

        return candidate_edges

def normalise_h(prob, weight, hnodes, bin_dim , indicator, edge_mask, indexlist):
    
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(np.multiply(prob, edge_mask), 10 * temp))
    
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10* temp))
    combined_problist = []
   
    problist = []
    for i in indexlist:
        for j in range(i+1, n):
            problist.append(p_rs[i][j])
            indi = np.multiply(indicator[i], indicator[j])
            denom = sum(np.multiply(w_rs[i][j], indi))
            if denom == 0:
                denom = 1
                del problist[-1]
            w_rs[i][j] = np.multiply(w_rs[i][j], indi)/ denom
            combined_problist.extend(p_rs[i][j]*w_rs[i][j])
    problist = np.array(problist)
    
    return combined_problist/problist.sum()

def checkcycle(edge, G=None):
    if G == None:
        G=nx.Graph()
    (u, v, w) = edge
    G.add_edge(u, v, weight=w)
    #return (G, len(list(nx.simple_cycles(G))))
    return (G, len(nx.cycle_basis(G)))

def log_fact(k):
    dict_ = defaultdict(float)
    for i in range(k):
        dict_[i+1] = dict_[i] + log(i+1)
    return dict_ 

def normalise(prob, weight, n, bin_dim, seen_list, list_edges, indicator):
    #'''
    #print "Debug", np.minimum(prob, np.zeros([n, n]).fill(10.0))
    n = len(prob[0])
    temp = np.ones([n, n])
    #print "Debug temp", temp
    #temp.fill(10.0)
    #print "Debug temp", np.minimum(prob, 10 * temp)

    p_rs = np.exp(np.minimum(prob, 10 * temp))
    p_rs = p_rs/p_rs.sum()
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    #w_rs = p_rs/p_rs.sum()
    combined_problist = []
    problist = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j,1) in seen_list or (i,j,2) in seen_list or (i,j,3) in seen_list:
                if (i, j, 1) in list_edges:
                    list_edges.remove((i, j, 1))
                if (i, j, 2) in list_edges:
                    list_edges.remove((i, j, 2))
                if (i, j, 3) in list_edges:
                    list_edges.remove((i, j, 3))
                continue

            problist.append(p_rs[i][j])
            indi = np.multiply(indicator[i], indicator[j])
            denom = sum(np.multiply(w_rs[i][j], indi))
            if denom == 0:
                denom = 1
                del problist[-1]
            w_rs[i][j] = np.multiply(w_rs[i][j], indi)/ denom
            combined_problist.extend(p_rs[i][j]*w_rs[i][j])
    problist = np.array(problist)
    #return problist/problist.sum(), list_edges, w_rs
    return combined_problist/problist.sum(), list_edges, w_rs


def get_candidate_edges(n):
    list_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # list_edges.append((i,j))
            list_edges.append((i, j, 1))
            list_edges.append((i, j, 2))
            list_edges.append((i, j, 3))
    return list_edges


def get_candidate_neighbor_edges(index, n):
    list_edges = []
    for j in range(n):
            if j == index:
                continue
            if j > index:
                #list_edges.append((index,j))
                list_edges.append((index, j, 1))
                list_edges.append((index, j, 2))
                list_edges.append((index, j, 3))
            else:
                #list_edges.append((j,index))
                list_edges.append((j, index, 1))
                list_edges.append((j, index, 2))
                list_edges.append((j, index, 3))
    
    return list_edges


def slerp(p0, p1, t):
    omega = arccos(dot(p0/norm(p0), p1/norm(p1)))
    so = sin(omega)
    if so == 0:
        return p0
    #print "Debug slerp", p0, p1, omega, so,  sin((1.0-t)*omega)/so,  sin((1.0-t)*omega)/so *np.array(p0)
    return sin((1.0-t)*omega) / so * np.array(p0) + sin(t*omega)/so * np.array(p1)


def lerp(p0, p1, t):
    return np.add(p0, t * np.subtract(p1,p0))


def degree(A):
    return np.zeros()


def construct_feed_dict(lr,dropout, k, n, d, decay, placeholders):
    # construct feed dictionary
    feed_dict = dict()


    #feed_dict.update({placeholders['features']: features})
    #feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['lr']: lr})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['decay']: decay})
    #feed_dict.update({placeholders['input']:np.zeros([k,n,d])})
    return feed_dict


def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()


def basis(adj, atol=1e-13, rtol=0):
    """Estimate the basis of a matrix.


    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    b : ndarray
        The basis of the columnspace of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
    """

    A = degree(adj) - adj

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    q, r = qr(A)
    return q[:rank]


def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))

def get_basis(mat):
    basis = np.zeros(1,1)
    return basis

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def pickle_load(path):
    '''Load the picke data from path'''
    with open(path, 'rb') as f:
        loaded_pickle = pickle.load(f)
    return loaded_pickle

def load_embeddings(fname, z_dim):
    embd = []
    with open(fname) as f:
        for line in f:
            embd.append(np.array(ast.literal_eval(line)).reshape((z_dim,1)))
    return embd

def load_data(filename, num=0, bin_dim=3):
    path = filename+"/*"
    adjlist = []
    featurelist = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    hdelist = []
    filenumber = int(len(glob.glob(path)) * 1)
    
    for fname in sorted(glob.glob(path))[:filenumber]:
        f = open(fname, 'r')
        try:
            G=nx.read_edgelist(f, nodetype=int)
        except:
            f = open(fname, 'r')
            lines = f.read()
            linesnew = lines.replace('{', '{\'weight\':').split('\n')
            G=nx.parse_edgelist(linesnew, nodetype=int)

            #print "Except"
            #continue
        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)

        # We assume there are only 4 types of atoms 
        degreemat = np.zeros((n, 4), dtype=np.float)
        #degreemat = np.zeros((n,1), dtype=np.float)
        count = np.zeros(4)

        for u in G.nodes():
            if G.degree(u) == 3 or G.degree(u) == 5:
                index = 2
            else:
                index = G.degree(u) -1
            degreemat[int(u)][index] = 1 
            #degreemat[int(u)][0] = (G.degree(u)*1.0)/(n-1)
            #count[G.degree(u)] += 1

        hde = (2 * count[3] + 2 + count[2] - count[0]) / 2
        hdelist.append(hde)
        
        try:
            weight = np.array(nx.adjacency_matrix(G).todense())
            adj = np.zeros([n,n])
            weight_bin = np.zeros([n,n,bin_dim])
            edges = []
            for i in range(n):
                for j in range(n):
                    if weight[i][j]>0:
                        adj[i][j] = 1
                        weight_bin[i][j][weight[i][j]-1] = 1
                        if j > i:
                            edges.append((i,j,weight[i][j]))
            adjlist.append(adj)
            weightlist.append(weight)
            weight_binlist.append(weight_bin)
            featurelist.append(degreemat)
            edgelist.append(edges)
        except:
            print("Error")
            continue
    
    return (adjlist, weightlist, weight_binlist, featurelist, edgelist, hdelist)


def pickle_save(content, path):
    '''Save the content on the path'''
    with open(path, 'wb') as f:
        pickle.dump(content, f)


def getedges(adj, n):
    edges = []
    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0:
                edges.append((i,j, adj[i][j]))
    return edges

from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops, MolFromSmiles, MolToSmiles
from rdkit import Chem
import sascorer
from checkvalidity import *

def compute_cost(G, writefile):
    cost = 0.0
    if guess_correct_molecules_from_graph(G, writefile):
        m1 = Chem.MolFromMol2File(sys.argv[2])
        if m1 != None:
            s = Chem.MolToSmiles(m1)
            sas = -sascorer.calculateScore(MolFromSmiles(s))
            logP = Descriptors.MolLogP(MolFromSmiles(s))
            cycle_list = nx.cycle_basis(G)

            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            cost = sas + logP + cycle_score
    # we want to define this property value such that low vales are better
    return -cost
