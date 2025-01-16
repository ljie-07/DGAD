import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='dataset name: Flickr/ACM/BlogCatalog/cora/citeseer/pubmed')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=200, help='Training epoch')
    parser.add_argument('--tests', type=int, default=1, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    # parser.add_argument('--alpha', type=float, default=1, help='balance parameter')
    parser.add_argument('--beta', type=float, default=0.4, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--patience', type=int, default=200, help='Patience')
    parser.add_argument('--weight_decay', type=float, default=2e-4)#0.0001
    parser.add_argument('--runs', type=int, default=10, help='Number of runs')
    args = parser.parse_args()
    return args