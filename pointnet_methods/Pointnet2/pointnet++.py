import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path

from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import sys
import importlib
import shutil
from data_utils.dataset import ShapeNetDataset
from sklearn.metrics import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=150, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["cpu_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    TRAIN_DATASET = ShapeNetDataset(train = True)

    TEST_DATASET = ShapeNetDataset(train= False)
    print(len(TRAIN_DATASET), len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 10
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0



    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    epoch_losses = []
    mean_acc = []
    total_correct = 0
    total_testset = 0
    epoch_acc = []
    epoch_ma_f1 = []
    epoch_precision = []
    epoch_recall = []
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0),total=len(trainDataLoader)):
            points, target = data
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        if epoch % 1 == 0:
            classifier = classifier.eval()
            test_pred, test_label = [], []
            total_correct = 0
            total_testset = 0
        with torch.no_grad():
            for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _ = classifier(points)
                pred_choice = pred.data.max(1)[1]
                target = target.long()
                correct = pred_choice.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
                pred = torch.max(pred, 1)[1].view(-1)
                test_pred += pred.detach().cpu().numpy().tolist()
                target =target.cpu()
                test_label += target.numpy().tolist()
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

    np.savetxt(r'F:\pointnet++\loss.txt', epoch_losses, fmt='%.4f')
    np.savetxt(r'F:\pointnet++\acc.txt', epoch_acc, fmt='%.4f')
    np.savetxt(r'F:\pointnet++_\precision.txt', epoch_precision, fmt='%.4f')
    np.savetxt(r'F:\pointnet++\recall.txt', epoch_recall, fmt='%.4f')
    np.savetxt(r'F:\pointnet++\ma_f1.txt', epoch_ma_f1, fmt='%.4f')
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
