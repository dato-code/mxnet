import numpy as np
from PIL import Image
import StringIO as _StringIO

import mxnet as mx
from rcnn.config import config
from rcnn.detector import Detector
from rcnn.symbol import get_vgg_test
from utils.load_model import load_param
from helper.processing import image_processing

try:
    import graphlab as gl
except:
    import sframe as gl

class VGGDetector(Detector):
    def __init__(self, prefix, epoch, ctx=mx.cpu()):
        sym = get_vgg_test()
        args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)
        super(VGGDetector, self).__init__(symbol=sym,
                                          ctx=ctx,
                                          arg_params=args,
                                          aux_params=auxs)
        self.classes = None
        config.TEST.HAS_RPN = True

    def detect(self, im, filter_result=True, column_name="image"):
        if type(im) == gl.SFrame:
            return self._detect_array(im["image"], filter_result)
        elif type(im) == gl.SArray:
            return self._detect_array(im, filter_result)
        elif type(im) == gl.Image:
            im_tensor, im_info = self.make_batch(im, True)
            return self.im_detect(im_tensor, im_info, filter_result)
        elif "PIL" in str(type(im)):
            im_tensor, im_info = self.make_batch(im, False)
            return self.im_detect(im_tensor, im_info, filter_result)
        else:
            raise Exception("Unknown input image type")

    def _detect_array(self, arr, filter_result):
        if filter_result == False:
            scores = []
            preds = []
            for im in arr:
                im_tensor, im_info = self.make_batch(im, True)
                score, pred = self.im_detect(im_tensor, im_info, False)
                scores.append(list(score))
                preds.append(list(pred))
            return gl.SFrame({"score" : scores, "bbox_pred" : preds})
        else:
            dets = []
            for im in arr:
                im_tensor, im_info = self.make_batch(im, True)
                det = self.im_detect(im_tensor, im_info, filter_result)
                dets.append(det)
            return gl.SFrame({"det" : dets})

    def make_batch(self, im, is_flex_image=True):
        if is_flex_image:
            img = Image.open(_StringIO.StringIO(im._image_data))
        else:
            img = im
        im, im_scale = image_processing.resize(img, 600, 1000)
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS)
        im_info = np.zeros((1, 3), dtype=np.float32)
        im_info[0, 0], im_info[0, 1] , im_info[0, 2] = im_tensor.shape[2], \
                                                       im_tensor.shape[3], \
                                                       im_scale
        return im_tensor, im_info


def create(model_type, prefix, epoch, ctx=mx.cpu()):
    if model_type == "vgg":
        return VGGDetector(prefix, epoch, ctx)


