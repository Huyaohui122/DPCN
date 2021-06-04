import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import Model10DataSet
from Dgcnn_model import  DGCNN
from torch.utils.data import DataLoader
from Dgcnn_util import cal_loss, IOStream
import sklearn.metrics as metrics
from sklearn.metrics import *
from tqdm import tqdm

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(Model10DataSet(train=True), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Model10DataSet(train=False), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)


    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=0.01)

    scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0

    epoch_losses = []
    mean_acc = []
    total_correct = 0
    total_testset = 0
    epoch_acc = []
    epoch_ma_f1 = []
    epoch_precision = []
    epoch_recall = []
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data)
            loss = criterion(logits, label.view(-1).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        epoch_losses.append(train_loss*1.0/count)
        ####################
        # Test
        ####################
        if epoch % 1 == 0:
            classifier = model.eval()
            test_pred, test_label = [], []
            total_correct = 0
            total_testset = 0
            with torch.no_grad():
                for i, data in tqdm(enumerate(test_loader, 0)):
                    points, target = data
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.to(device), target.to(device)

                    pred = classifier(points)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data.long()).to(device).sum()
                    total_correct += correct.item()
                    total_testset += points.size()[0]
                    pred = torch.max(pred, 1)[1].view(-1)
                    test_pred += pred.detach().cpu().numpy().tolist()
                    test_label += target.cpu().numpy().tolist()

            print('correct_cnt: {}, total_cnt: {}'.format(total_correct, total_testset))
            acc = total_correct / total_testset
            epoch_acc.append(acc)
            print('Accuracy: {:.4f}'.format(acc))
            precision = precision_score(test_label, test_pred, average='macro')
            recall = recall_score(test_label, test_pred, average='macro')

            ma_f1 = f1_score(test_label, test_pred, average='macro')
            epoch_ma_f1.append(ma_f1)
            epoch_precision.append(precision)
            epoch_recall.append(recall)
            print('precision: {:.4f}'.format(precision))
            print('recall: {:.4f}'.format(recall))
            print('ma_f1: {:.4f}'.format(ma_f1))
            print("final accuracy {}".format(total_correct / float(total_testset)))

    np.savetxt(r'F:\DGCNN\loss.txt', epoch_losses, fmt='%.4f')
    np.savetxt(r'F:\DGCNN\acc.txt', epoch_acc, fmt='%.4f')
    np.savetxt(r'F:\DGCNN\precision.txt', epoch_precision, fmt='%.4f')
    np.savetxt(r'F:\DGCNN\recall.txt', epoch_recall, fmt='%.4f')
    np.savetxt(r'F:\DGCNN\ma_f1.txt', epoch_ma_f1, fmt='%.4f')






if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='test_5010', metavar='N',
                        choices=['test_5010'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use Adam')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=16,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=4, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)

