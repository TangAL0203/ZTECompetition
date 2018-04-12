#-*-coding:utf-8-*-
import onnx
import caffe2.python.onnx.backend as backend
from caffe2.python.onnx.backend import Caffe2Backend as c2
# load onnx object
model = onnx.load("./squeezenet1_1.onnx")
prepared_backend = backend.prepare(model)
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model.graph)

try:
    with open("squeeze_init_net.pb", "wb") as f:
        f.write(init_net.SerializeToString())
    f.close()
    with open("squeeze_predict_net.pb", "wb") as f:
        f.write(predict_net.SerializeToString())
    f.close()
except:
    print("not okokok")


