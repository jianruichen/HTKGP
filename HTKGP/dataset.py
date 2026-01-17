# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import torch
import math
import copy
import time
import numpy as np
from random import shuffle
from manifolds.poincare import PoincareBall
from sklearn.metrics.pairwise import cosine_similarity

from scripts import shredFacts
from collections import Counter

class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, 
                 ds_name):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.tim2id = {}
        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.start_batch = 0
        self.all_facts_as_tuples = None
        self.trans=PoincareBall()
        self.convertTimes()
        
        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])
        
    def readFile(self, 
                 filename):

        with open(filename, "r",encoding="utf-8") as f:
             data = f.readlines()
        
        facts = []
        for line in data:
            elements = line.strip().split("\t")
            
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            timestamp = elements[3]
            tim_id = self.getTimID(elements[3])
            facts.append([head_id, rel_id, tail_id, tim_id,timestamp])
            
        return facts
    
    
    def convertTimes(self):      
        """
        This function spits the timestamp in the day,date and time.
        """  
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date
    
    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)
        
    def numTim(self):

        return len(self.tim2id)

    
    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name] 
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]
        
        
    def getTimID(self, tim_name):
        if tim_name in self.tim2id:
            return self.tim2id[tim_name]
        self.tim2id[tim_name] = len(self.tim2id)
        return self.tim2id[tim_name]

    
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch : ]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts
    

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio
            
            facts[s1:e1,0] = (facts[s1:e1,0] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            facts[s2:e2,2] = (facts[s2:e2,2] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            
        return facts
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)
    
    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio))
        return batch
    
    
    def wasLastBatch(self):
        return (self.start_batch == 0)

    def Entropy(self,X,counts):
        counter = Counter(X)
        a = sorted(counter.items(), key=lambda x: x[0], reverse=False)
        b = dict(a)
        prob = {i[0]: i[1] / counts for i in b.items()}
        H = {i[0]:-i[1] * math.log2(i[1]) for i in prob.items()}
        return list(H.values())


    def getadj(self, data,device): #simple graph
        H1 = torch.zeros((len(self.ent2id), len(self.ent2id))).to(device)
        a = torch.tensor(data[:, 1]).long().to(device)
        b = torch.tensor(data[:, 0]).long().to(device)
        c = torch.tensor(data[:, 2]).long().to(device)
        for i, m in zip(a,b):
            H1[i, m] = 1
        for i, m in zip(a, c):
            H1[i, m] = 1
        H2=torch.mm(H1,H1).to(device)
        H3 = H1 + H2+torch.eye(H1.size(0),H1.size(0)).to(self.device)
        self.H=H1
        D = torch.sum(H3, dim=0)
        D = torch.tensor(D, dtype=torch.float)
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        edge_index = (H3 / D_sqrt)
        adj = (edge_index / D_sqrt.t())
        normalize0 = torch.nn.init.normal(adj, mean=0, std=1)
        ecos=self.trans.hyperbolic_distance(normalize0,normalize0,c=0.5).to(device=self.device)
        # ecos = cosine_similarity(normalize0.cpu(), normalize0.cpu())
        ecos=ecos.pow(2)
        # ecos = torch.tensor(ecos).to(device=self.device)
        return adj+ecos


    def getHadj(self, data,device):
        counts = len(data)
        X = []
        for i in range(counts):
            X.append(data[i][0])
        pro = self.Entropy(X, counts)
        pro = torch.tensor(pro).to(device)
        del X
        H1 = torch.zeros((len(self.rel2id), len(self.ent2id))).to(device)
        H4 = torch.zeros((len(self.tim2id), len(self.ent2id))).to(device)
        a = torch.tensor(data[:, 1]).long().to(device)
        b = torch.tensor(data[:, 0]).long().to(device)
        c = torch.tensor(data[:, 2]).long().to(device)
        d = torch.tensor(data[:, 3]).long().to(device)
        l = torch.stack([b, c], dim=0).to(device)
        for j in l:
            for i, m in zip(a, j):
                if m == 0:
                    continue
                else:
                    H1[i, m] = 1

            for i, m in zip(d, j):
                if m == 0:
                    continue
                elif H4[i,m] == 1:
                    H4[i, m] += 1
                else:
                    H4[i, m] = 1
        T = torch.zeros((len(self.tim2id), len(self.tim2id))).to(device)
        R = torch.zeros((len(self.rel2id), len(self.rel2id))).to(device)
        W = torch.zeros((len(self.rel2id), len(self.rel2id))).to(device)
        for i in range(0, len(self.rel2id) - 1):
            W[i + 1, i + 1] = pro[i]
        RH = torch.cat([R, H1], dim=1).to(device)
        HE = torch.cat([torch.mm(H1.T, W), self.H], dim=1).to(device)
        RE = torch.cat([RH, HE], dim=0).to(device)
        RED = torch.sum(RE, dim=1).to(device)
        RED[RED == 0.] = 1.
        RED_sqrt = torch.sqrt(RED).unsqueeze(dim=0).to(device)
        RE1 = (RE / RED_sqrt).to(device)
        graph0 = (RE1 / RED_sqrt.t()).to(device)
        del RH, HE, RED,RE1, RED_sqrt, pro, data

        TH = torch.cat([T, H4], dim=1).to(device)
        HE = torch.cat([H4.T, self.H], dim=1).to(device)
        O = torch.cat([TH, HE], dim=0).to(device)
        OD = torch.sum(O, dim=1).to(device)
        OD[OD == 0.] = 1.
        OD_sqrt = torch.sqrt(OD).unsqueeze(dim=0).to(device)
        O1 = (O / OD_sqrt).to(device)
        graph3 = (O1 / OD_sqrt.t()).to(device)
        del TH, HE, OD, OD_sqrt, O1
        return graph0, graph3
