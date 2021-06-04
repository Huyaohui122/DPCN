from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import *
def parse_args():
    "parameters"
    parser = argparse.ArgumentParser("pointnet")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    return parser.parse_args()

def main(args):
    blue = lambda x: '\033[94m' + x + '\033[0m'

    args.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            train = True)

        test_dataset = ShapeNetDataset(
           train= False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=int(args.workers))

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=int(args.workers))

    print(len(dataset), len(test_dataset))
    num_classes = 10
    print('classes', num_classes)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=args.feature_transform)

    if args.model != '':
        classifier.load_state_dict(torch.load(args.model))
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.to(device)

    num_batch = len(dataset) / args.batchSize

    epoch_losses = []
    epoch_acc = []
    epoch_ma_f1 = []
    epoch_precision = []
    epoch_recall = []
    for epoch in range(1 ,args.nepoch+1):
        print('Epoch-{} testing'.format(epoch))
        scheduler.step()

        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            target =target.long()
            loss = F.cross_entropy(pred, target)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            classifier = classifier.eval()
            test_pred, test_label = [], []
            total_correct = 0
            total_testset = 0
            with torch.no_grad():
                for i, data in tqdm(enumerate(testdataloader, 0)):
                  points, target = data
                  target = target[:, 0]
                  points = points.transpose(2, 1)
                  points, target = points.to(device), target.to(device)
                  target =target.long()
                  pred, _, _ = classifier(points)
                  pred_choice = pred.data.max(1)[1]
                  correct = pred_choice.eq(target.data).to(device).sum()
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


    np.savetxt(r'F:\pointnet\loss.txt', epoch_losses, fmt='%.4f')
    np.savetxt(r'F:\pointnet\acc.txt', epoch_acc, fmt='%.4f')
    np.savetxt(r'F:\pointnet\precision.txt', epoch_precision, fmt='%.4f')
    np.savetxt(r'F:\pointnet\recall.txt', epoch_recall, fmt='%.4f')
    np.savetxt(r'F:\pointnet\ma_f1.txt', epoch_ma_f1, fmt='%.4f')

if __name__ == '__main__':
    args = parse_args()
    main(args)
