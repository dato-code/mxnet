import mxnet as mx
import numpy as np
import cv2


from rcnn.config import config
from helper.processing.bbox_transform import bbox_transform_inv, clip_boxes
from helper.processing.nms import nms
from helper.processing import image_processing

class Detector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.arg_params = arg_params
        if aux_params == None:
            self.aux_params = {}
        self.aux_params = aux_params
        self.executor = None

    def im_detect(self, im_array, im_info, get_feature=False):
        """
        perform detection of designated im, box, must follow minibatch.get_testbatch format
        :param im_array: numpy.ndarray [b c h w]
        :param im_info: numpy.ndarray [b 3]
        :return: scores, pred_boxes
        """
        self.arg_params['data'] = mx.nd.array(im_array, self.ctx)
        self.arg_params['im_info'] = mx.nd.array(im_info, self.ctx)
        arg_shapes, out_shapes, aux_shapes = \
            self.symbol.infer_shape(data=self.arg_params['data'].shape,
                                    im_info=self.arg_params['im_info'].shape)
        arg_shapes_dict = {name: shape for name, shape in zip(self.symbol.list_arguments(), arg_shapes)}
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}
        self.arg_params['cls_prob_label'] = mx.nd.zeros(arg_shapes_dict['cls_prob_label'], self.ctx)
        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)
        output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}

        self.executor.forward(is_train=False)
        scores = output_dict['cls_prob_output'].asnumpy()
        bbox_deltas = output_dict['bbox_pred_output'].asnumpy()
        rois = output_dict['rpn_rois_rpn_rois'].asnumpy()
        num_classes = scores.shape[1]
        feature = None
        if get_feature == True:
            feature = output_dict['feature_output'].asnumpy()
        return self._filter_result(num_classes, scores, bbox_deltas, rois, im_info, feature)

    def _filter_result(self, num_classes, scores, box_deltas, rois, im_info, feature, thresh=0.05):
        result = {}
        boxes = rois[:, 1:5] / im_info[0][2]
        boxes = bbox_transform_inv(boxes, box_deltas)
        boxes = clip_boxes(boxes, (im_info[0][0] / im_info[0][2],
                                   im_info[0][1] / im_info[0][2],
                                   3))

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            if feature is not None:
                fea = feature[inds, :]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, config.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if feature is not None:
                fea = fea[keep, :]
            if cls_dets.shape[0] == 0:
                continue
            result[j] = {}
            result[j]["dets"] = cls_dets.copy()
            result[j]["features"] = fea
        return result

    def get_image_array(self, path, scales=600):
        """
        build image array from specific roidb
        :param roidb: images to be processed
        :param scales: scale list
        :param scale_indexes: indexes
        :return: array [b, c, h, w], list of scales
        """
        im = cv2.imread(path)
        target_size = scales
        im_info = np.zeros((1, 3), dtype="float32")
        im, im_scale = image_processing.resize(im, target_size, 1000)
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS)
        im_info[0, 0], im_info[0, 1] = im_tensor.shape[-2:]
        im_info[0, 2] = im_scale
        return im_tensor, im_info

