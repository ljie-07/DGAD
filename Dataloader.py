import numpy as np
import scipy.sparse as sp
from utils import  *
from scipy.io import loadmat
import torch
from utils import normalize,sparse_mx_to_torch_sparse_tensor
from args import parameter_parser
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix

args = parameter_parser()

def load_mat(dataset, datadir='dataset'):
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network'] if ('Network' in data_mat) else data_mat['A']
    feat = data_mat['Attributes'] if ('Attributes' in data_mat) else data_mat['X']
    truth = data_mat['Label'] if ('Label' in data_mat) else data_mat['gnd']
    # feat =feat.toarray()
    if args.dataset == 'dgraphfin':
     test_id = data_mat['k'].flatten()
    else:
     test_id = None
    # feat = feat.toarray()
    # str_truth = data_mat['str_anomaly_label']
    # attr_truth = data_mat['attr_anomaly_label']
    truth = truth.flatten()


    if args.dataset == 'Books' or args.dataset == 'Enron':
        adj = csc_matrix(adj)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = normalize_adj(adj)

    # adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    # adj = adj.toarray()


    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # dis_adj = sparse_mx_to_torch_sparse_tensor(dis_adj)


    # feat = feat.toarray()
    feat = sp.lil_matrix(feat).toarray()
    # feat = torch.FloatTensor(feat)


    # sht_x = generate_negative(adj,feat)
    # dis_adj = normalize_adj(dis_adj)
    # dis_adj = dis_adj.toarray()


    # return adj_norm, feat, truth, adj ,str_truth ,attr_truth
    return adj_norm, feat, truth, adj,test_id


