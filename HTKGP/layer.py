import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.weight = Parameter(torch.FloatTensor(in_ft, out_ft),requires_grad=True).to(self.device)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft),requires_grad=True).to(self.device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, Graph):
        support = torch.mm(input, self.weight).to(self.device)
        output = torch.spmm(Graph, support).to(self.device)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    
class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
         super(HGNN_classifier, self).__init__()
         self.fc = nn.Linear(n_hid, n_class)

    def forward(self, x):
         x = self.fc(x)
         return x
        

