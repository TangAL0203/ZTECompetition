#-*-coding:utf-8-*-
import io
import numpy as np
import torch.onnx
import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import sys
import logging
import caffe2
import models


torch_model = models.squeezenet1_1(20, pretrained=False)
state = torch.load('./squeezenet1_10.766.pth')
torch_model.load_state_dict(state)

from torch.autograd import Variable
batch_size = 1    # just a random number

# Input to the model
x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "squeezenet1_1.onnx",       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file


# Convert to pb
import onnx
import onnx_caffe2.backend
# load onnx object
model = onnx.load("squeezenet1_1.onnx")
prepared_backend = onnx_caffe2.backend.prepare(model)
from onnx_caffe2.backend import Caffe2Backend as c2
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model.graph)
with open("squeeze_init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open("squeeze_predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())
