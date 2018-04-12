#-*-coding:utf-8-*-
from torch.autograd import Variable
import torchvision
import onnx
import numpy as np
import torch
import models
import caffe2

'''
1. 这里之所以要设置一个输入x，因为onnx采用 track机制，会先随便拿个符合输入size的数据跑一遍，拿到网络结构
2. 转完后先生成onnx object： mobilenetv2.onnx
3. tutorials link: https://github.com/onnx/tutorials/blob/master/tutorials/PytorchCaffe2SuperResolution.ipynb
'''
model = models.MobileNetV2(20)
state = torch.load('./mobilenetv20.722.pth')
model.load_state_dict(state, strict=False)

# model = torchvision.models.squeezenet1_1(pretrained=True)

batch_size=1  # 随便一个数
x = Variable(torch.randn(batch_size,3,224,224), requires_grad=True)
torch_out = torch.onnx._export(model,
                              x,
                              "./squeezenet1_1.onnx",
                              export_params=True
                              )
print("okokok")