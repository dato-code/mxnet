import sys
sys.path.insert(0, "../../python")

import mxnet as mx
import numpy as np
import cv2


from rcnn.detector import Detector
from rcnn.symbol import get_symbol_vgg_test
from tools.load_model import load_param

sym = get_symbol_vgg_test(num_classes=81, scales=[4,8,16,32], get_feature=True)
arg_params, aux_params = load_param("./mscoco", 1, ctx=mx.gpu(1))

det = Detector(symbol=sym, ctx=mx.gpu(1), arg_params=arg_params, aux_params=aux_params)
im_array, im_info = det.get_image_array("/home/bing/detection/py-faster-rcnn/data/demo/004545.jpg")

result = det.im_detect(im_array, im_info, get_feature=True)

import pickle
pickle.dump(result, open("sample.pkl", "wb"))
