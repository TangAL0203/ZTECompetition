#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable
import utils.dataset as dataset
import torchvision.datasets.folder as folder
from utils.BasicTrainTest import *
import utils.models as models
import warnings
from time import time
warnings.filterwarnings('ignore')

use_gpu = torch.cuda.is_available()

root = './test/'
path = './test/沙发/10.jpg'

classes, class_to_idx = folder.find_classes('./test/')

idx_to_class = {}

for classname, idx in class_to_idx.items():
    idx_to_class[idx] = classname

checkpointPath = './checkpoint/squeezenet1_10.766.pth'
state = torch.load(checkpointPath)
model = models.squeezenet1_1(20, pretrained=False)
if use_gpu:
    model = model.cuda()
model.load_state_dict(state)

model.eval()

img, label = dataset.get_single_img(root, path, train=False)
if use_gpu:
    output = model(Variable(img.cuda()))
else:
    output = model(Variable(img))

pred_label = int(output.data.max(1)[1].cpu())

for idx, name in idx_to_class.items():
    if idx==int(label):
        print("true label is: ")
        print name
    if idx==pred_label:
        print("pred label is: ")
        print name



