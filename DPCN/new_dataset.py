import os
import torch.utils.data as data
#using the rotating building data...
#bu_filename = 'test_5010r'
import geopandas as gpd
from shapely import affinity
from shapely.geometry import Polygon
import numpy as np
import copy
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
    , 'seq_length': 32  #
    , 'rotate_type': 'equal'  #
    , 'rotate_length': 1  #
    , 'norm_type': 'minmax'  #

    # model params
    , 'GPU': True  #
    , 'epochs': 400  #
    , 'optimizer': 'rmsprop'  # 均方根传播法
    , 'z_size': 128  #
    , 'rnn_type': 'lstm'  # lstm or gru
}
bu_shape = gpd.read_file('./data/' + bu_filename + '.shp', encode='utf-8')
bu_use = copy.deepcopy(bu_shape)  # deepcopy的结果不会随原数据的改变而改变
bu_mbr, bu_use = get_shape_mbr(bu_use)
bu_use = get_shape_normalize_final(bu_use, hparams['if_scale_y'])  # hparams在55行自定义参数 False
if hparams['if_simplify']:  # True
    bu_use = get_shape_simplify(bu_use, hparams['tor_dist'], hparams['tor_cos'], simplify_type=0)  # 0.1 0.99
bu_use = reset_start_point(bu_use)  # 重设多边形起始点
bu_node = get_node_features(bu_use)
bu_line = get_line_features_final(bu_node, hparams['seq_length'])
bu_detail = get_inter_features(bu_line)  # 得到插入特征
# outputpath = "./data/bu_detail.csv"
# bu_detail.to_csv(outputpath, sep=',', index=True, header=True)
bu_detail = get_neat_features(bu_detail, hparams['seq_length'], hparams['rotate_length'])  # 处理，得到平滑特征
# outputpath = "./data/bu_detail_neat.csv"
# bu_detail.to_csv(outputpath, sep=',', index=True, header=True)
test_x = np.array(bu_detail['xs']).reshape(5010*hparams['seq_length'], 1)
test_y = np.array(bu_detail['ys']).reshape(5010*hparams['seq_length'], 1)
test_x_y = np.concatenate((test_x, test_y), axis=1)
test_xy_reshape = test_x_y.reshape(5010, hparams['seq_length'], 2)

class ModelNet40DataSet(data.Dataset):
    def __init__(self, train=True):
        label = np.genfromtxt("data/label.content", dtype=np.int64).reshape(-1,1)
        modelnet_data = np.zeros([0,hparams['seq_length'],2], dtype=np.float64)
        modelnet_label = np.zeros([0,1], np.float64)
        modelnet_data = np.concatenate([modelnet_data, test_xy_reshape],axis=0)
        modelnet_label = np.concatenate([modelnet_label,label],axis=0)
        self.point_cloud = modelnet_data
        self.label = modelnet_label
    def __getitem__(self, item):
        return self.point_cloud[item], self.label[item]

    def __len__(self):
        return self.label.shape[0]




if __name__ == '__main__':
    modelnet_train = ModelNet40DataSet(train=True)
    modelnet_test = ModelNet40DataSet(train=False)

    data, label = modelnet_train[0]
    print('type: {}'.format(type(data)))
    print('type: {}'.format(type(label)))
    print(label)