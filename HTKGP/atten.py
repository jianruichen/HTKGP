import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.stats import norm
import torch.nn as nn
from torch.nn.parameter import Parameter

class GATLayer(nn.Module):
    def __init__(self,edim):
        super(GATLayer, self).__init__()
        self.a=Parameter(torch.FloatTensor(edim,1), requires_grad=True)
        self.a.data.uniform_(-(1/edim), (1/edim))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, c0,c1,c2,c3,c4):
        a = F.tanh(torch.mm(c0,self.a).to(self.device))
        a1 = F.tanh(torch.mm(c1,self.a).to(self.device))
        a2 = F.tanh(torch.mm(c2,self.a).to(self.device))
        a3 = F.tanh(torch.mm(c3,self.a).to(self.device))
        a4 = F.tanh(torch.mm(c4,self.a).to(self.device))
        a_exp = torch.exp(a).to(self.device)
        a1_exp = torch.exp(a1).to(self.device)
        a2_exp = torch.exp(a2).to(self.device)
        a3_exp = torch.exp(a3).to(self.device)
        a4_exp = torch.exp(a4).to(self.device)
        a = (a_exp / (a_exp + a1_exp+a2_exp+ a3_exp+a4_exp)).to(self.device)
        a1 = (a1_exp / (a_exp + a1_exp+a2_exp+ a3_exp+a4_exp )).to(self.device)
        a2 = (a2_exp / (a_exp + a1_exp+a2_exp+ a3_exp+a4_exp )).to(self.device)
        a3 = (a3_exp / (a_exp + a1_exp+a2_exp+ a3_exp+a4_exp )).to(self.device)
        a4 = (a4_exp / (a_exp + a1_exp+a2_exp+ a3_exp+a4_exp )).to(self.device)
        return a,a1,a2,a3,a4

