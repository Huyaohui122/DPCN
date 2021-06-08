from numpy import *
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import warnings
warnings.filterwarnings("ignore")
from dataset import Model10DataSet
from Model.DPCN import DPCN_vanilla
from params import Args
import matplotlib.pyplot as plt
from numpy import *
from time import strftime, localtime
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    pwd = os.getcwd()
    weights_dir = os.path.join(pwd, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    logging.info('Loading Dataset...')
    train_dataset = Model10DataSet(train=True)
    test_dataset = Model10DataSet(train=False)
    logging.info('train_dataset: {}'.format(len(train_dataset)))
    logging.info('test_dataset: {}'.format(len(test_dataset)))
    logging.info('Done...\n')


    logging.info('Creating DataLoader...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Args.batch_size, shuffle=False, num_workers=2)
    logging.info('Done...\n')


    logging.info('Checking gpu...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info('gpu available: {}'.format(torch.cuda.device_count()))
        logging.info('current gpu: {}'.format(torch.cuda.get_device_name(0)))
        logging.info('gpu capability: {}'.format(torch.cuda.get_device_capability(0)))
    else:
        logging.info('gpu not available, running on cpu instead.')
    logging.info('Done...\n')


    logging.info('Create SummaryWriter in ./summary')
    summary_writer = SummaryWriter(comment='DPCN', log_dir='summary')
    logging.info('Done...\n')
    logging.info('Creating Model...')
    model = DPCN_vanilla(num_classes=10).to(Args.device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedular = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    logging.info('Done...\n')
    epoch_losses = []
    epoch_acc = []
    logging.info('Start training...')
    epoch_losses = []
    epoch_ma_f1 = []
    epoch_precision = []
    epoch_recall = []
    for epoch in range(1, Args.num_epochs+1):
        logging.info("--------Epoch {}--------".format(epoch))
        schedular.step()
        tqdm_batch = tqdm(train_loader, desc='Epoch-{} training'.format(epoch))

        # train
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        model.train()
        loss_tracker = AverageMeter()
        for batch_idx, (data, label) in enumerate(tqdm_batch):
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = criterion(out, label.view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.update(loss.item(), label.size(0))


        tqdm_batch.close()
        logging.info('Loss: {:.4f} ({:.4f})'.format(loss_tracker.val, loss_tracker.avg))

        summary_writer.add_scalar('loss', loss_tracker.avg, epoch)
        epoch_losses.append(loss_tracker.avg)

        if epoch % Args.test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))

            model.eval()
            test_pred, test_label = [], []
            correct_cnt = 0
            total_cnt = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(tqdm_batch):
                    data, label = data.to(device), label.to(device)
                    out = model(data)
                    pred_choice = out.max(1)[1]
                    label = label.long()
                    correct_cnt += pred_choice.eq(label.view(-1)).sum().item()
                    total_cnt += label.size(0)
                    pred = torch.max(out, 1)[1].view(-1)
                    test_pred += pred.detach().cpu().numpy().tolist()
                    test_label += label.cpu().numpy().tolist()

            print('correct_cnt: {}, total_cnt: {}'.format(correct_cnt, total_cnt))
            acc = correct_cnt / total_cnt
            logging.info('Accuracy: {:.4f}'.format(acc))
            epoch_acc.append(acc)
            summary_writer.add_scalar('acc', acc, epoch)
            precision = precision_score(test_label, test_pred, average='macro')
            recall = recall_score(test_label, test_pred, average='macro')
            ma_f1 = f1_score(test_label, test_pred, average='macro')
            epoch_ma_f1.append(ma_f1)
            epoch_precision.append(precision)
            epoch_recall.append(recall)
            print('precision: {:.4f}'.format(precision))
            print('recall: {:.4f}'.format(recall))
            print('ma_f1: {:.4f}'.format(ma_f1))
            tqdm_batch.close()


        if epoch % Args.save_freq == 0:
            ckpt_name = os.path.join(weights_dir, 'DPCN_{0}.pth'.format(epoch))
            torch.save(model.state_dict(), ckpt_name)
            logging.info('model saved in {}'.format(ckpt_name))
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    summary_writer.close()

    np.savetxt(r'F:\DPCN\loss.txt',epoch_losses,fmt='%.4f')
    np.savetxt(r'F:\DPCN\acc.txt', epoch_acc,fmt='%.4f')
    np.savetxt(r'F:\DPCN\precision.txt', epoch_precision,fmt='%.4f')
    np.savetxt(r'F:\DPCN\recall.txt', epoch_recall,fmt='%.4f')
    np.savetxt(r'F:\DPCN\ma_f1.txt', epoch_ma_f1,fmt='%.4f')

if __name__ == '__main__':
    train()
