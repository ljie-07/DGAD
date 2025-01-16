
from layers import *
from utils import *

args = parameter_parser()

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x



class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x



class Fusion(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Fusion, self).__init__()
        self.fusion = nn.Linear(hidden_size*2, hidden_size*2)

        self.hidden_size = hidden_size
        self.dropout = dropout

    def forward(self, x_1, x_lin):
        combined_H = torch.cat((x_1, x_lin), dim=1)
        imp = self.fusion(combined_H)
        imp = torch.tanh(imp)
        imp = torch.mean(imp, dim=0)
        x_f = imp[:self.hidden_size] * x_1 + imp[self.hidden_size:] * x_lin

        return x_f



class DGAD(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(DGAD, self).__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.lin = nn.Linear(feat_size, hidden_size)
        self.fusion = Fusion(hidden_size, dropout)


    def forward(self, x, adj):
        x_1 = self.shared_encoder(x, adj)
        x_lin = self.lin(x)
        x_2 = self.fusion(x_1, x_lin)
        x_hat = self.attr_decoder(x_2, adj)

        return x_hat, x_lin, x_1,x_2






