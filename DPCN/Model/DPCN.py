import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchsummary import summary
from collections import OrderedDict
from params import Args
def conv_bn_block(input, output, kernel_size):
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class TriangleConv(nn.Module):
    def __init__(self, layers):
        super(TriangleConv, self).__init__()
        self.layers = layers
        mlp_layers = OrderedDict()
        for i in range(len(self.layers) - 1):
            if i == 0:
                mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(4 * self.layers[i], self.layers[i + 1], 1)
            else:
                mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(self.layers[i], self.layers[i + 1], 1)
        self.mlp = nn.Sequential(mlp_layers)



    def forward(self, X):
        B, N, F = X.shape
        k_indexes = []
        for i in range(N):
            if i == 0:
                k_indexes.append([N - 1, i + 1])
            elif i == N-1:
                k_indexes.append([i - 1, 0])
            else:
                k_indexes.append([i - 1, i+1])
        k_indexes_tensor = torch.Tensor(k_indexes)
        k_indexes_tensor = k_indexes_tensor.long()
        x1 = torch.zeros(B, N, 2, F).to(Args.device)
        for idx, x in enumerate(X):
            x1[idx] = x[k_indexes_tensor]
        x2 = X.reshape([B, N, 1, F]).float()
        x2 = x2.expand(B, N, 2, F)
        x2 = x2-x1
        x3 = x2[:, :, 0:1, :]
        x4 = x2[:, :, 1:2, :]
        x4 = x3-x4
        x5 = X.reshape([B, N, 1, F]).float()
        x2 = x2.reshape([B, N, 1, 2*F])
        x_triangle = torch.cat([x5, x2, x4], dim=3)
        x_triangle=torch.squeeze(x_triangle)
        x_triangle = x_triangle.permute(0, 2, 1)
        x_triangle = torch.tensor(x_triangle,dtype=torch.float32).to(Args.device)
        out = self.mlp(x_triangle)
        out = out.permute(0, 2, 1)
        return out


class DPCN_vanilla(nn.Module):
    def __init__(self, num_classes):
        super(DPCN_vanilla, self).__init__()

        self.num_classes = num_classes
        self.triangleconv_1 = TriangleConv(layers=[2, 64, 64, 64])
        self.triangleconv_2 = TriangleConv(layers=[64, 512,1024])
        self.fc_block_4 = fc_bn_block(1024, 512)
        self.drop_4 = nn.Dropout(0.5)
        self.fc_block_5 = fc_bn_block(512, 256)
        self.drop_5 = nn.Dropout(0.5)
        self.fc_6 = nn.Linear(256, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        assert C == 2, 'dimension of x does not match'
        x = self.triangleconv_1(x)
        x = self.triangleconv_2(x)
        x = x.permute(0, 2, 1)
        x = nn.MaxPool1d(N)(x)
        x = x.reshape([B, 1024])
        x = self.fc_block_4(x)
        x = self.drop_4(x)
        x = self.fc_block_5(x)
        x = self.drop_5(x)
        x = self.fc_6(x)
        x = F.log_softmax(x, dim=-1)

        return x












