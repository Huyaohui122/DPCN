from time import strftime, localtime

import shapefile  # shapefile库安装：pip install pyshp
import numpy as np
import dgl
import torch
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
from numpy import *
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim import lr_scheduler
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import torch.nn as nn
import torch.nn.functional as F

from new_pre import *
bu_filename = 'test_5010'
hparams = {
    'if_simplify': True
    , 'tor_dist': 0.1
    , 'tor_cos': 0.99
    , 'if_scale_y': False
    , 'seq_length': 16
    , 'rotate_length': 1
}
bu_shape = gpd.read_file('./data/' + bu_filename + '.shp', encode='utf-8')
bu_use = copy.deepcopy(bu_shape)
bu_mbr, bu_use = get_shape_mbr(bu_use)
bu_use = get_shape_normalize_final(bu_use, hparams['if_scale_y'])
if hparams['if_simplify']:
    bu_use = get_shape_simplify(bu_use, hparams['tor_dist'], hparams['tor_cos'], simplify_type=0)
bu_use = reset_start_point(bu_use)
bu_node = get_node_features(bu_use)
bu_line = get_line_features_final(bu_node, hparams['seq_length'])
bu_detail = get_inter_features(bu_line)
bu_detail = get_neat_features(bu_detail, hparams['seq_length'], hparams['rotate_length'])


all_data_x = np.array(bu_detail['xs'])
all_data_y = np.array(bu_detail['ys'])


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
    point_x = []
    point_y = []
    for idx in  range(point):
        point_x.append(all_data_x[all_data_range].reshape(-1,1))
        point_y.append(all_data_y[all_data_range].reshape(-1,1))
        all_data_range=all_data_range+1

    total_feature = np.concatenate((point_x, point_y),axis=1)
    total_feature = total_feature.squeeze()

    g.ndata['x'] = torch.tensor(total_feature, dtype=torch.float64)
    graph.append(g)


label = np.genfromtxt("label.content", dtype=np.float64)


totalset = []
for i in range(len(graph)):
    temp = (graph[i], label[i])
    totalset.append(temp)


trainset = totalset[:4000]
testset = totalset[4000 :]

pwd = os.getcwd()
weights_dir = os.path.join(pwd, 'weights')


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

import torch.optim as optim
from torch.utils.data import DataLoader


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)
test_loader = DataLoader(testset, batch_size=32, shuffle=False,
                         collate_fn=collate)

model = Classifier(2, 512, 10)
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
        ma_f1 = f1_score(test_label,test_pred,average='macro')
        epoch_ma_f1.append(ma_f1)
        epoch_precision.append(precision)
        epoch_recall.append(recall)
        print('precision: {:.4f}'.format(precision))
        print('recall: {:.4f}'.format(recall))
        print('ma_f1: {:.4f}'.format(ma_f1))
    if epoch % 1== 0:
        ckpt_name = os.path.join(weights_dir, 'GCN-2point_{0}.pth'.format(epoch))
        torch.save(model.state_dict(), ckpt_name)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))


# plt.plot(epoch_losses)
# plt.show()
np.savetxt(r'F:\Users\GCN-2point\loss.txt',epoch_losses,fmt='%.4f')
np.savetxt(r'F:\Users\GCN-2point\_acc.txt', epoch_acc,fmt='%.4f')
np.savetxt(r'F:\Users\GCN-2point\precision.txt', epoch_precision,fmt='%.4f')
np.savetxt(r'F:\Users\GCN-2point\_recall.txt', epoch_recall,fmt='%.4f')
np.savetxt(r'F:\Users\GCN-2point\ma_f1.txt', epoch_ma_f1,fmt='%.4f')
