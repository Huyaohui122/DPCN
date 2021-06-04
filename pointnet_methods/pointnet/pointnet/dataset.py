from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
from new_pre import *
bu_filename = 'test_5010'     #'test_5010r'
hparams = {
    # preprocessing params
    # 曲线变直线方法参数
    'if_simplify': True
    , 'tor_dist': 0.1
    , 'tor_cos': 0.99
    , 'if_scale_y': False

    , 'scale_type': 'final'
    , 'seq_length':  16 #
    , 'rotate_type': 'equal'  #
    , 'rotate_length': 1  #
    , 'norm_type': 'minmax'  #
}

class ShapeNetDataset(data.Dataset):
    def __init__(self, train=True):
        bu_shape = gpd.read_file('../new/' + bu_filename + '.shp', encode='utf-8')
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
        index = 4000
        all_data_x = np.array(bu_detail['xs'])
        train_x = all_data_x[:index * hparams['seq_length']]
        train_x = train_x.reshape(index * hparams['seq_length'], 1)
        test_x = all_data_x[index * hparams['seq_length']:]
        test_x = test_x.reshape(-1, 1)

        all_data_y = np.array(bu_detail['ys'])
        train_y = all_data_y[:index * hparams['seq_length']]
        train_y = train_y.reshape(index * hparams['seq_length'], 1)
        test_y = all_data_y[index * hparams['seq_length']:]
        test_y = test_y.reshape(-1, 1)
        train_x_y = np.concatenate((train_x, train_y), axis=1)
        test_x_y = np.concatenate((test_x, test_y), axis=1)
        train_xy_reshape = train_x_y.reshape(index, hparams['seq_length'], 2)
        test_xy_reshape = test_x_y.reshape(5010 - index, hparams['seq_length'], 2)

        label = np.genfromtxt("../new/label.content", dtype=np.int64)
        label_train = label[:index]
        label_train = label_train.reshape(-1, 1)
        label_test = label[index:]
        label_test = label_test.reshape(-1, 1)
        if train:
            modelnet_data = np.zeros([0, hparams['seq_length'], 2], dtype=np.float64)
            modelnet_label = np.zeros([0, 1], np.float64)
            modelnet_data = np.concatenate([modelnet_data, train_xy_reshape], axis=0)
            modelnet_label = np.concatenate([modelnet_label, label_train], axis=0)
        else:
            modelnet_data = np.zeros([0, hparams['seq_length'], 2], dtype=np.float64)
            modelnet_label = np.zeros([0, 1], np.float64)
            modelnet_data = np.concatenate([modelnet_data, test_xy_reshape], axis=0)
            modelnet_label = np.concatenate([modelnet_label, label_test], axis=0)

        self.point_cloud = modelnet_data
        self.label = modelnet_label
    def __getitem__(self, item):
        return self.point_cloud[item], self.label[item]

    def __len__(self):
        return self.label.shape[0]


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)


