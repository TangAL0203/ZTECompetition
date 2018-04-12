#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision
import torchvision.datasets.folder as folder
import math
import os
import getpass
import shutil
import argparse
import utils.models as models
from utils.BasicTrainTest import *
import warnings
warnings.filterwarnings('ignore')

use_gpu = torch.cuda.is_available()
num_batches = 0

if getpass.getuser() == 'tsq':
    train_batch_size = 8
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_batch_size = 64

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch finetune mobilenetv1 or v2')
    parser.add_argument('--arch', metavar='ARCH', default='mobilenetv1', help='model architecture')
    parser.add_argument('--checkpoint', default=False, action='store_true', help='choose if train from checkpoint')
    args = parser.parse_args()
    return args

args = get_args()
arch = args.arch
checkpoint = args.checkpoint

print("arch is: ",arch)
print("checkpoint is: ",checkpoint)

def finetune():
    global train_batch_size
    if arch == "mobilenetv1":
        model = models.MobileNet(20)
        if use_gpu:
            model = model.cuda()
    elif arch == "mobilenetv2":
        model = models.MobileNetV2(20)
        if use_gpu:
            model = model.cuda()
    elif arch == 'squeezenet1_1':
        model = models.squeezenet1_1(20, pretrained=False)
        if use_gpu:
            model = model.cuda()

    if checkpoint:
        if arch == "mobilenetv2":
            checkpointPath = './checkpoint/mobilenetv20.865.pth'
            state = torch.load(checkpointPath)
            model.load_state_dict(state)
            optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        if arch == 'squeezenet1_1':
            checkpointPath = './checkpoint/squeezenet1_10.762.pth'
            state = torch.load(checkpointPath)
            model.load_state_dict(state)
            optimizer = optim.SGD(model.newclassifier.parameters(), lr = 0.01, momentum=0.9)
    else:
        if arch == "mobilenetv1":
            state = torch.load('./checkpoint/mobilenet_sgd_rmsprop_69.526.tar')['state_dict']
            model.load_state_dict(state, strict=False)
            optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        elif arch == "mobilenetv2":
            state = torch.load('./checkpoin t/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar')
            model.load_state_dict(state, strict=False)
            optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        elif arch == 'squeezenet1_1':
            model = models.squeezenet1_1(20, pretrained=True)
            if use_gpu:
                model = model.cuda()
            optimizer = optim.SGD(model.newclassifier.parameters(), lr = 0.01, momentum=0.9)

    # train_path = '/home/smiles/ModelPruning/CatDog/train'
    # test_path = '/home/smiles/ModelPruning/CatDog/test'

    train_path = './train'
    # test_path = train_path
    test_path = './test'
    train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)

    train_test('./checkpoint/', arch, model, train_loader, test_loader, optimizer=optimizer, epoches=40)

if __name__ == '__main__':
    finetune()