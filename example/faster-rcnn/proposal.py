# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# Modification by Bing Xu
# --------------------------------------------------------

import sys
sys.path.insert(0, "../../python/")

import mxnet as mx
import numpy as np

from helper.config import cfg
from helper.generate_anchors import generate_anchors
from helper.bbox_transform import bbox_transform_inv, clip_boxes
from helper.nms_wrapper import nms


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

class ProposalLayer(mx.operator.NumpyOp):
    def __init__(self, feat_stride, scales, is_train=False):
        super(ProposalLayer, self).__init__(need_top_grad=False)
        self._feat_stride = feat_stride
        anchor_scales = scales
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        if is_train == True:
            self.cfg_key = 'TRAIN'
        else:
            self.cfg_key = 'TEST'


    def list_arguments(self):
        return ['rpn_cls_prob', 'rpn_bbox_pred', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        cfg_key = self.cfg_key
        rpn_cls_prob_shape = in_shape[0]
        rpn_bbox_pred_shape = in_shape[1]
        assert(rpn_cls_prob_shape[0] == rpn_bbox_pred_shape[0])
        if rpn_cls_prob_shape[0] > 1:
            raise ValueError("Only single item batches are supported")

        batch_size = rpn_cls_prob_shape[0]
        im_info_shape = (batch_size, 3)

        output_shape = (1, cfg[cfg_key].RPN_POST_NMS_TOP_N, 5)
        return [rpn_cls_prob_shape, rpn_bbox_pred_shape, im_info_shape], [output_shape]

    def forward(self, in_data, out_data):
        # assume need to get context in future
        cfg_key = self.cfg_key
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = in_data[0][:, self._num_anchors:, :, :]
        bbox_deltas = in_data[1]
        im_info = in_data[2][0, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel())).transpose()


        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))


        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        out_data[0][:] = blob.reshape((1, blob.shape[0], blob.shape[1]))
        # top[0].reshape(*(blob.shape))
        # top[0].data[...] = blob

        # [Optional] output scores blob
        # if len(top) > 1:
        #     top[1].reshape(*(scores.shape))
        #     top[1].data[...] = scores


    def backward(self, out_grad, in_data, out_data, in_grad):
        """This layer does not propagate gradients."""
        pass

