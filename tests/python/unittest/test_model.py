import mxnet as mx
import numpy as np
from common import models

def test_get_feature_symbol():
    data = mx.sym.Variable("data")
    conv1_1 = mx.sym.Convolution(data=data, kernel=(3,3), pad=(1,1), num_filter=10, name="conv1_1")
    conv1_2 = mx.sym.Convolution(data=data, kernel=(5,5), pad=(2,2), num_filter=20, name="conv1_2")
    concat = mx.sym.Concat(*[conv1_1, conv1_2])
    conv2 = mx.sym.Convolution(data=concat, kernel=(3,3), pad=(1,1), num_filter=40, name="conv2")
    conv3 = mx.sym.Convolution(data=conv2, kernel=(3,3), pad=(1,1), num_filter=30, name="conv3")
    flatten = mx.sym.Flatten(data=conv3, name="flatten")
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=100)
    softmax = mx.sym.SoftmaxOutput(data=fc)

    model = mx.model.FeedForward(symbol=softmax)
    fea_sym = mx.model.get_feature_symbol(model, "flatten_output")
    assert fea_sym.list_arguments() == flatten.list_arguments()

    fea_sym = mx.model.get_feature_symbol(model)
    assert fea_sym.list_arguments() == flatten.list_arguments()

if __name__ == '__main__':
    test_get_feature_symbol()

