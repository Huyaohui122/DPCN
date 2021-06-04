from time import strftime, localtime

import shapefile  # shapefile库安装：pip install pyshp
import numpy as np
import dgl
import torch
from numpy import *
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import torch.utils.data as data
#using the rotating building data...
#bu_filename = 'test_5010r'
import geopandas as gpd
from shapely import affinity
from shapely.geometry import Polygon
import numpy as np
import copy
from sklearn.metrics import f1_score
from preprocess import *
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import *
from preprocess import *
bu_filename = 'test_5010'
hparams = {

    'if_simplify': True
    , 'tor_dist': 0.1
    , 'tor_cos': 0.99
    , 'if_scale_y': False

    , 'scale_type': 'final'
    , 'seq_length': 16  #
    , 'rotate_type': 'equal'  #
    , 'rotate_length': 1  #
    , 'norm_type': 'minmax'  #


}
k_list=[2, 4]
bu_shape=gpd.read_file('./data/' + bu_filename + '.shp', encode='utf-8')
bu_use=copy.deepcopy(bu_shape)
bu_mbr,bu_use=get_shape_mbr(bu_use)
bu_use=get_shape_normalize_final(bu_use,hparams['if_scale_y'])
if hparams['if_simplify']:
    bu_use=get_shape_simplify(bu_use,hparams['tor_dist'],hparams['tor_cos'],simplify_type=0)
bu_use = reset_start_point(bu_use)
bu_geos = np.array([geo for geo in bu_use[['geometry']].values])
bu_node=get_node_features(bu_use)
bu_line=get_line_features_final(bu_node, hparams['seq_length'])
bu_detail=get_inter_features(bu_line)
bu_detail=get_neat_features(bu_detail, hparams['seq_length'], hparams['rotate_length'])
bu_features=get_multi_features_final(bu_detail, bu_use,k_list)
bu_features=get_overall_features_final(bu_features, bu_use)
bu_features=get_normalize_features_final(bu_features, hparams['norm_type'])

cols=[
    'k1_l_bc','k1_s_abc','k1_s_obc','k1_c_obc','k1_r_obc','k1_rotate_bac','k1_rotate_boc'
    ,'k2_l_bc','k2_s_abc','k2_s_obc','k2_c_obc','k2_r_obc','k2_rotate_bac','k2_rotate_boc'
    ,'l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius'
]
cols=[
    c for c in bu_features.columns if 'k' in c
]+['Elongation','Circularity','Rectangularity','Convexity','l_oa','MeanRedius']

bu_seq = get_train_sequence(bu_features,cols,hparams['rotate_type'])
featureList=[x for x in bu_seq.columns if 'f_' in x]
de_input_list,en_input_list,de_target_list=get_seq2seq_train_dataset(bu_seq)
dataset_en_input=np.array([[timestamps for timestamps in sample] for sample in bu_seq[en_input_list].values])
dataset_de_input=np.array([[timestamps for timestamps in sample] for sample in bu_seq[de_input_list].values])
dataset_de_output=np.array([[timestamps for timestamps in sample] for sample in bu_seq[de_target_list].values])
bu_x = np.array([[timestamps for timestamps in sample] for sample in bu_seq[en_input_list].values])
all_data_range=0
graph = []
for i in range(5010):
    point = 16
    uu = []
    vv = []
    for j in range(point):
        uu.append(j)
        if j + 1 in range(point):
            vv.append(j+1)
        else:
            vv.append(0)
    u = np.concatenate([uu, vv])
    v = np.concatenate([vv, uu])
    g = dgl.graph((torch.tensor(uu), torch.tensor(vv)))
    g =dgl.to_bidirected(g)
    g.edges()
    g.ndata['x'] = torch.tensor( bu_x[(hparams['rotate_length']+0)*i].tolist(), dtype=torch.float64)
    graph.append(g)
label = np.genfromtxt("label.content", dtype=np.float64)
totalset = []
for i in range(len(graph)):
    temp = (graph[i], label[i])
    totalset.append(temp)
trainset = totalset[:4000]
testset = totalset[4000 :]
print(len(trainset))
print(len(testset))

pwd = os.getcwd()
weights_dir = os.path.join(pwd, 'weights')

from torch.optim import lr_scheduler
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 256)
        self.classify = nn.Linear(256, n_classes)

    def forward(self, g):

        h = g.ndata['x'].float()
        h = F.leaky_relu(self.conv1(g, h))
        h = F.leaky_relu(self.conv2(g, h))

        g.ndata['x'] = h
        hg = dgl.mean_nodes(g, 'x')
        return self.classify(hg)




def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)
test_loader = DataLoader(testset, batch_size=32, shuffle=False,
                         collate_fn=collate)

model = Classifier(20, 512, 10)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
schedular = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
model.train()
epoch_losses = []
epoch_acc = []
epoch_ma_f1 = []
epoch_precision = []
epoch_recall = []
for epoch in range(150):
    schedular.step()
    epoch_loss = 0
    for iter, (batchg, label) in enumerate(data_loader):
        prediction = model(batchg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

    model.eval()
    test_pred, test_label = [], []
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            pred = torch.softmax(model(batchg), 1)
            pred_choice = pred.max(1)[1]
            correct_cnt += pred_choice.eq(label.view(-1)).sum().item()
            total_cnt += label.size(0)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
        print('correct_cnt: {}, total_cnt: {}'.format(correct_cnt, total_cnt))
        acc = correct_cnt / total_cnt
        epoch_acc.append(acc)
        print('Accuracy: {:.4f}'.format(acc))
        precision = precision_score(test_label,test_pred,average='macro')
        recall = recall_score(test_label,test_pred,average='macro')
        confusion_matrix_new = confusion_matrix(test_label,test_pred)
        ma_f1 = f1_score(test_label,test_pred,average='macro')
        epoch_ma_f1.append(ma_f1)
        epoch_precision.append(precision)
        epoch_recall.append(recall)
        print('precision: {:.4f}'.format(precision))
        print('recall: {:.4f}'.format(recall))
        print('ma_f1: {:.4f}'.format(ma_f1))
    if epoch % 1== 0:
        ckpt_name = os.path.join(weights_dir, 'GCN_{0}.pth'.format(epoch))
        torch.save(model.state_dict(), ckpt_name)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
np.savetxt(r'F:\Users\GCN\loss.txt',epoch_losses,fmt='%.4f')
np.savetxt(r'F:\Users\GCN\_acc.txt', epoch_acc,fmt='%.4f')
np.savetxt(r'F:\Users\GCN\precision.txt', epoch_precision,fmt='%.4f')
np.savetxt(r'F:\Users\GCN\_recall.txt', epoch_recall,fmt='%.4f')
np.savetxt(r'F:\Users\GCN\ma_f1.txt', epoch_ma_f1,fmt='%.4f')
