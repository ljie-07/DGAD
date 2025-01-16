import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
import scipy.io as sio
import random
from sklearn.metrics import precision_recall_curve,auc
# import dgl
import math
from sklearn import manifold
from sklearn.neighbors import kneighbors_graph
from args import parameter_parser
import torch.nn.functional as F
from torch.sparse import mm
import os.path as osp
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
args = parameter_parser()



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # sp_tuple = sparse_to_tuple(features)
    if sp.issparse(features):
        features = features.todense()
    return features




def add_gaussian_noise(features, mean=0.0, std=1.0):
    noise = torch.randn(features.size()) * std + mean
    noise = noise.to(features.device)
    noisy_features = features + noise
    return noisy_features


def generate_negative(adj, node_features):
    num_nodes = node_features.shape[0]
    neg_feature = np.zeros_like(node_features)

    for i in range(num_nodes):
        no_neighbors = np.where(adj[0][i] == 0)

        if len(no_neighbors) > 0:
            neg_feature[i] = np.mean(node_features[no_neighbors], axis=0, dtype=np.float32)

    return neg_feature




def calculate_auprc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    return auprc

def remove_edges(adj, p):
    # Remove a fraction p of edges from the adjacency matrix
    adj_triu = sp.triu(adj)  # get upper triangle of adjacency matrix
    edges = adj_triu.nonzero()  # get indices of upper triangle edges
    # randomly remove a fraction p of edges
    n_remove = int(p * len(edges[0]))
    # adj_array = adj.toarray()
    adj_array = adj

    num = 0
    while num < n_remove:
        edge_index = np.random.randint(len(edges[0]))
        if adj_array[edges[0][edge_index]].sum() > 1.2 and adj_array[edges[1][edge_index]].sum() > 1.2:
            adj_array[edges[0][edge_index]][edges[1][edge_index]] = 0
            adj_array[edges[1][edge_index]][edges[0][edge_index]] = 0
            num += 1
    adj_new = adj_array
    # adj_new = sp.csr_matrix(adj_array)
    # count = count_changed_adj_elements(adj, adj_new)
    return adj_new





def correlation_coefficient(x, y):

    cov = torch.mean((x - torch.mean(x)) * (y - torch.mean(y)))

    std_x = torch.std(x)
    std_y = torch.std(y)

    corr = cov / (std_x * std_y)
    return corr



def negative_sampling(raw_adj):

    device = raw_adj.device
    adj = raw_adj.coalesce()
    indices = adj.indices()
    N = raw_adj.size(0)
    num_positive = indices.size(1)+1 //2

    i = torch.min(indices[0], indices[1])
    j = torch.max(indices[0], indices[1])
    existing_edges = (i * N + j).unique()
    existing_edges, _ = existing_edges.sort()

    negative_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    while negative_edges.size(1) < num_positive:
        remaining = num_positive - negative_edges.size(1)
        batch_size = remaining * 2


        sampled_i = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
        sampled_j = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)


        mask = sampled_i < sampled_j
        sampled_i = sampled_i[mask]
        sampled_j = sampled_j[mask]


        sampled_encoded = sampled_i * N + sampled_j
        sampled_encoded_sorted, _ = sampled_encoded.sort()


        positions = torch.searchsorted(existing_edges, sampled_encoded_sorted)
        mask_not_exist = (positions >= existing_edges.size(0)) | (existing_edges[positions] != sampled_encoded_sorted)
        sampled_i = sampled_i[mask_not_exist]
        sampled_j = sampled_j[mask_not_exist]

        if sampled_i.numel() == 0:
            continue


        new_neg = torch.stack([sampled_i, sampled_j], dim=0)
        negative_edges = torch.cat([negative_edges, new_neg], dim=1)



    negative_edges = negative_edges[:, :num_positive]

    if negative_edges.size(1) == 0:

        return torch.sparse_coo_tensor(indices, torch.ones(num_positive, dtype=raw_adj.dtype, device=device),
                                       raw_adj.size())


    neg_values = torch.ones(num_positive, dtype=raw_adj.dtype, device=device)
    neg_adj = torch.sparse_coo_tensor(negative_edges, neg_values, raw_adj.size()).coalesce()

    return neg_adj






