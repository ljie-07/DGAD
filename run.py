import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import GAD
import torch_geometric.utils as utils
from utils import *
from args import parameter_parser
from Dataloader import load_mat
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,auc,precision_score,precision_recall_curve
import torch.nn.functional as F
import math
import random
import os
# import dgl
import argparse
from tqdm import tqdm
import numpy as np
import json
from utils import *


def max_message(feature, adj_matrix):
    if adj_matrix.is_sparse:
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
    else:
        row_sum = adj_matrix.sum(dim=1)

    r_inv = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))

    feature_normalized = torch.nn.functional.normalize(feature, p=2, dim=1, eps=1e-30)

    if adj_matrix.is_sparse:
        agg_features = torch.sparse.mm(adj_matrix, feature_normalized)
    else:
        agg_features = torch.matmul(adj_matrix, feature_normalized)

    message = (feature_normalized * agg_features).sum(dim=1)

    message = message * r_inv

    return message




#loss function
def loss_func(attrs, X_hat):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    # diff_structure = torch.pow(A_hat - adj, 2)
    # structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    # structure_cost = torch.mean(structure_reconstruction_errors)
    # cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    cost = attribute_reconstruction_errors

    return cost,  attribute_cost

args = parameter_parser()
print(args)

# Load and preprocess data
# adj, features, label, adj_label ,str_label ,attr_label = load_mat(args.dataset)
adj, features, label, adj_label,test_id= load_mat(args.dataset)

if args.dataset in ['Amazon']:
    features = preprocess_features(features)
if args.dataset in ['Reddit']:
    features = (features - features.mean(0)) / (features.std(0) +1e-30)

features = torch.FloatTensor(features)
nb_nodes = features.shape[0]
seeds = [i for i in range(args.runs)]
all_auc = []
all_auprc = []
for run in range(args.runs):
    seed = seeds[run]
    print('\n# Run:{}'.format(run), flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loss_values = []
    aucc = []

    cnt_wait = 0
    best = 1e9
    best_t = 0

    patience = args.patience
    max_score = 0.
    b_xent = nn.BCEWithLogitsLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    feat_size = features.size(1)


    model = GAD(feat_size=features.size(1), hidden_size=args.hidden_dim, dropout=args.dropout)
    if args.device == 'cuda' and torch.cuda.is_available():
        print('use cuda')
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        features = features.to(device)
        model = model.to(device)


    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_train = time.time()

    for epoch in range(args.epoch):

        model.train()
        optimiser.zero_grad()

        X_hat, x_lin, x_1,x_2 = model(features, adj)
        mem_train = torch.cuda.max_memory_allocated()



        sc1 = max_message(x_2, adj_label)

        dis_adj = negative_sampling(adj_label)

        sc2 = max_message(x_2,  dis_adj)

        loss, feat_loss = loss_func(features, X_hat)

        pred = torch.cat([sc1, sc2], dim=0)


        lbl_1 = torch.ones(nb_nodes)
        lbl_2 = torch.zeros(nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 0).to(device)

        b_xent = nn.BCEWithLogitsLoss(reduction='none')
        loss_2 = b_xent(pred,lbl)
        loss2 = torch.mean(loss_2)

        loss3 = -torch.abs(correlation_coefficient(x_lin , x_2))

        loss3 = torch.mean(loss3)

        l = torch.mean(loss)*args.beta+loss2*(1-args.beta)+loss3

        if  l < best:
            best = l
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        l.backward()
        optimiser.step()

        loss_tu = l.item()
        loss_values.append(loss_tu)

        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()))
    time_train = time.time() - time_train
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pth'))
    multi_round_ano_score = np.zeros((args.tests, nb_nodes))
    torch.cuda.reset_peak_memory_stats()
    for test in range(args.tests):

            with torch.no_grad():
                X_hat, x_lin, x_1,x_2 = model(features, adj)
                mem_test = torch.cuda.max_memory_allocated()


                sc1 = max_message(x_2, adj_label)



                loss, feat_loss = loss_func(features, X_hat)

                score1 = -sc1
                score1 = score1.detach().cpu().numpy()
                score2 = loss.detach().cpu().numpy()

                scaler1 = MinMaxScaler()
                scaler2 = MinMaxScaler()

                score1 = scaler1.fit_transform(score1.reshape(-1, 1)).reshape(-1)
                score2 = scaler2.fit_transform(score2.reshape(-1, 1)).reshape(-1)


            score = score2*args.beta + score1*(1-args.beta)

            ano_score_final_tran = np.transpose(score)


            multi_round_ano_score[test] = score


            auc_score = roc_auc_score(label, score)



            aucc.append(auc_score)
            if (auc_score > max_score):
                max_score = auc_score

            print("Epoch:", '%04d' % (test), 'Auc:', auc_score, 'Best_Auc:', max_score)


    ano_score_final = np.mean(multi_round_ano_score, axis=0)

    print("time_train:", time_train)
    print("mem_train:", mem_train/1024/1024)
    print("mem_test:", mem_test/1024/1024)

    auc_score = roc_auc_score(label, ano_score_final)
    auprc = calculate_auprc(label, ano_score_final)
    print("auprc:", auprc)
    print("auc:",auc_score)
    all_auc.append(auc_score)
    all_auprc.append(auprc)
print('\n==============================')
print(all_auc)
print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)),'FINAL TESTING AUC std:{:.4f}'.format(np.std(all_auc)))
print('{:.4f} ({:.4f})'.format(np.mean(all_auc),np.std(all_auc)))
print(all_auprc)
print('FINAL TESTING AUPRC:{:.4f}'.format(np.mean(all_auprc)),'FINAL TESTING AUPRC std:{:.4f}'.format(np.std(all_auprc)))
print('{:.4f} ({:.4f})'.format(np.mean(all_auprc),np.std(all_auprc)))
print('==============================')


args_dict = vars(args)

# Create results dictionary with timestamp
from datetime import datetime
results = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'args': args_dict,
    'all_auc': all_auc,
    'all_auprc': all_auprc,
    'final_test_auc_mean': np.mean(all_auc),
    'final_test_auc_std': np.std(all_auc),
    'final_test_auprc_mean': np.mean(all_auprc),
    'final_test_auprc_std': np.std(all_auprc)
}

# Write results to a text file
with open('results.txt', 'a') as f:
    f.write(json.dumps(results, indent=4))
    f.write('\n\n')

