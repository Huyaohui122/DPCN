from __future__ import print_function
import torch.utils.data as data
from new_pre import *
bu_filename = 'test_5010'
hparams = {
    'if_simplify': True
    , 'tor_dist': 0.1
    , 'tor_cos': 0.99
    , 'if_scale_y': False

    , 'scale_type': 'final'
    , 'seq_length':  16
    , 'rotate_type': 'equal'
    , 'rotate_length': 1
    , 'norm_type': 'minmax'
}


class Model10DataSet(data.Dataset):
    def __init__(self, train=True):
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
        index =4000
        all_data_x = np.array(bu_detail['xs'])
        train_x = all_data_x[:index*hparams['seq_length']]
        train_x = train_x.reshape(index*hparams['seq_length'], 1)
        test_x = all_data_x[index*hparams['seq_length']:]
        test_x = test_x.reshape(-1, 1)

        all_data_y = np.array(bu_detail['ys'])
        train_y = all_data_y[:index*hparams['seq_length']]
        train_y = train_y.reshape(index*hparams['seq_length'], 1)
        test_y = all_data_y[index*hparams['seq_length']:]
        test_y = test_y.reshape(-1, 1)
        train_x_y = np.concatenate((train_x, train_y), axis=1)
        test_x_y = np.concatenate((test_x, test_y), axis=1)
        train_xy_reshape = train_x_y.reshape(index, hparams['seq_length'], 2)
        test_xy_reshape = test_x_y.reshape(5010-index, hparams['seq_length'], 2)

        label = np.genfromtxt("data/label.content", dtype=np.int64)
        label_train =label[:index]
        label_train =label_train.reshape(-1,1)
        label_test = label[index:]
        label_test = label_test.reshape(-1,1)
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




if __name__ == '__main__':
    modelnet_train = Model10DataSet(train=True)
    modelnet_test = Model10DataSet(train=False)

    data, label = modelnet_train[0]
    print('type: {}'.format(type(data)))
    print('type: {}'.format(type(label)))
    print(label)