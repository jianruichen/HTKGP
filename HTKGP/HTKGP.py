# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import manifolds
from layer import HGNN_conv,HGNN_classifier
from atten import GATLayer
from hyplayers1 import *

class HTKGP(torch.nn.Module):
    def __init__(self, dataset, params):
        super(HTKGP, self).__init__()
        self.dataset = dataset
        self.params = params
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim + params.t_emb_dim).to(self.device)
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim + params.t_emb_dim).to(self.device)
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.s_emb_dim + params.t_emb_dim).to(self.device)
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.s_emb_dim + params.t_emb_dim).to(self.device)
        self.tim_embs = nn.Embedding(dataset.numTim(), params.emb_dim).to(self.device)
        self.create_time_embedds()
        self.time_nl = torch.sin
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
        nn.init.xavier_uniform_(self.tim_embs.weight)
        self.manifold = getattr(manifolds, params.mainfolds)()
        # self.A=kneighbors_graph(self.ent_embs_h.weight,5,mode='conectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)
        self.adj= self.dataset.getadj(self.dataset.data["train"],  device=self.device)
        self.hadj, self.hadj1 = self.dataset.getHadj(self.dataset.data["train"], device=self.device)
        self.hgc1 = HGCNConv1(self.manifold, params.emb_dim, params.emb_dim, dropout=0.3, act=F.leaky_relu)
        self.hgc2 = HGCNConv1(self.manifold, params.emb_dim, params.emb_dim, dropout=0.3, act=F.leaky_relu)
        self.hgc3 = HGCNConv1(self.manifold, params.emb_dim, params.emb_dim, dropout=0.3, act=F.leaky_relu)
        self.fc0 = HGNN_classifier(params.emb_dim, params.emb_dim)
        self.fc = HGNN_classifier(dataset.numEnt() + dataset.numTim(), params.emb_dim)
        self.fc1 = HGNN_classifier(dataset.numRel() + dataset.numTim(), params.emb_dim)
        self.fc2 = HGNN_classifier(dataset.numRel() + dataset.numEnt(), params.emb_dim)
        self.att = GATLayer(params.emb_dim)

    def create_time_embedds(self):
        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).to(self.device)

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t, c):
        if h_or_t == "head":
            emb = self.y_amps_h.weight[entities] * self.time_nl(self.y_freq_h.weight[entities] * years + self.y_phi_h.weight[entities])
            emb += self.m_amps_h.weight[entities] * self.time_nl(self.m_freq_h.weight[entities] * months + self.m_phi_h.weight[entities])
            emb += self.d_amps_h.weight[entities] * self.time_nl(self.d_freq_h.weight[entities] * days + self.d_phi_h.weight[entities])
        else:
            emb = self.y_amps_t.weight[entities] * self.time_nl(self.y_freq_t.weight[entities] * years + self.y_phi_t.weight[entities])
            emb += self.m_amps_t.weight[entities] * self.time_nl(self.m_freq_t.weight[entities] * months + self.m_phi_t.weight[entities])
            emb += self.d_amps_t.weight[entities] * self.time_nl(self.d_freq_t.weight[entities] * days + self.d_phi_t.weight[entities])
        emb = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(emb, c), c),c)
        return emb

    def getEmbeddings(self, rels,heads,  tails, years, months, days, emb,eemb,c,intervals=None):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)
        h_embs1 = emb[:, 0:self.params.s_emb_dim][heads]
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = eemb[:, 0:self.params.s_emb_dim][tails]
        h_embs2 = emb[:, 0:self.params.s_emb_dim][tails]
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = eemb[:, 0:self.params.s_emb_dim][heads]
        h_embs1 = torch.cat((h_embs1, (emb[:, self.params.s_emb_dim:][heads])+((self.get_time_embedd(heads, years, months, days, "head",c)))), 1)
        t_embs1 = torch.cat((t_embs1, (eemb[:, self.params.s_emb_dim:][tails])+((self.get_time_embedd(tails, years, months, days, "tail",c)))), 1)
        h_embs2 = torch.cat((h_embs2, (emb[:, self.params.s_emb_dim:][tails])+((self.get_time_embedd(tails, years, months, days, "head",c)))), 1)
        t_embs2 = torch.cat((t_embs2, (eemb[:, self.params.s_emb_dim:][heads])+((self.get_time_embedd(heads, years, months, days, "tail",c)))), 1)
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def computer0(self, e,c):
        x_tan = self.manifold.proj_tan0(e, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        e_emb = (self.hgc1(x_hyp, self.adj,c)).to(self.device)
        e_emb0 = (self.hgc1(e_emb, self.adj,c)).to(self.device)
        del x_hyp,x_tan,e_emb
        return e_emb0

    def computer1(self, r, e, c):
        re = torch.cat([r, e], dim=0)
        x_tan = self.manifold.proj_tan0(re, c).to(self.device)
        x_hyp = self.manifold.expmap0(x_tan, c).to(self.device)
        x_hyp = self.manifold.proj(x_hyp, c).to(self.device)
        rembs = [x_hyp]
        re_emb = (self.hgc2(x_hyp, self.hadj, c)).to(self.device)
        rembs.append(re_emb)
        re_emb1 = (self.hgc2(re_emb, self.hadj, c)).to(self.device)
        rembs.append(re_emb1)
        rembs = torch.stack(rembs, dim=1).to(self.device)
        rembs = torch.mean(rembs, dim=1).to(self.device)
        rm, em = torch.split(rembs, [self.dataset.numRel(), self.dataset.numEnt()])
        return em,rm

    def computer2(self, t, e, c):
        te = torch.cat([t, e], dim=0)
        x_tan = self.manifold.proj_tan0(te, c).to(self.device)
        x_hyp = self.manifold.expmap0(x_tan, c).to(self.device)
        x_hyp = self.manifold.proj(x_hyp, c).to(self.device)
        tembs = [x_hyp]
        te_emb = (self.hgc3(x_hyp, self.hadj1, c)).to(self.device)
        tembs.append(te_emb)
        te_emb1 = (self.hgc3(te_emb, self.hadj1, c)).to(self.device)
        tembs.append(te_emb1)
        tembs = torch.stack(tembs, dim=1).to(self.device)
        tembs = torch.mean(tembs, dim=1).to(self.device)
        tm, em = torch.split(tembs, [self.dataset.numTim(), self.dataset.numEnt()])
        return em,tm

    def cov(self):
        c0 = F.leaky_relu(self.fc(torch.cat([self.ent_embs_h.weight, self.tim_embs.weight], dim=0).T))
        c1 = F.leaky_relu(self.fc(torch.cat([self.ent_embs_t.weight, self.tim_embs.weight], dim=0).T))
        c2 = F.leaky_relu(self.fc1(torch.cat([self.rel_embs_f.weight, self.tim_embs.weight], dim=0).T))
        c3 = F.leaky_relu(self.fc1(torch.cat([self.rel_embs_i.weight, self.tim_embs.weight], dim=0).T))
        c4 = F.leaky_relu(self.fc2(torch.cat([self.rel_embs_i.weight + self.rel_embs_f.weight, self.ent_embs_h.weight * self.ent_embs_t.weight],dim=0).T))
        a, a1, a2, a3, a4 = self.att(c0, c1, c2, c3, c4)
        a.view(self.params.emb_dim)
        a1.view(self.params.emb_dim)
        a2.view(self.params.emb_dim)
        a3.view(self.params.emb_dim)
        a4.view(self.params.emb_dim)
        cc = (torch.einsum('ik,ij->ij', a, c0) + torch.einsum('ik,ij->ij', a1, c1) + torch.einsum('ik,ij->ij', a2,c2) + torch.einsum('ik,ij->ij', a3, c3) + torch.einsum('ik,ij->ij', a4, c4)).to(self.device)
        c = 1 / torch.abs(torch.sum((self.fc0(cc))))
        del c1,c2,c3,c4,c0
        e0 = self.computer0(self.ent_embs_h.weight,c)
        e00,r00=self.computer1(self.rel_embs_f.weight * self.rel_embs_i.weight,self.ent_embs_h.weight,c)
        e000,t000=self.computer2(self.tim_embs.weight,self.ent_embs_h.weight,c)
        e = self.ent_embs_h.weight + e0+e00+e000
        del e0,e00,e000
        # e_emb = F.dropout(e, p=self.params.edropout, training=self.training)
        e1 = self.computer0(self.ent_embs_t.weight,c)
        e11, r11 = self.computer1(self.rel_embs_f.weight * self.rel_embs_i.weight, self.ent_embs_t.weight,c)
        e111, t111 = self.computer2(self.tim_embs.weight, self.ent_embs_t.weight,c)
        ee = self.ent_embs_t.weight + e1+e11+e111
        del e1,e11,e111
        # ee_emb = F.dropout(ee, p=self.params.edropout, training=self.training)
        return e,ee,c

    def forward(self,  rels,heads, tails, years, months, days):
        emb,eemb,c = self.cov()
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(rels,heads, tails, years, months, days,emb,eemb,c)
        score = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        score = F.dropout(score, p=self.params.dropout, training=self.training)
        score = torch.sum(score, dim=1)
        return score


