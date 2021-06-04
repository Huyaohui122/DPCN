import os
import torch

class Args:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 检测是否可以使用gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    num_epochs = 150
    test_freq = 1
    save_freq = 1
    weight_reg = 0.01
    K = 4