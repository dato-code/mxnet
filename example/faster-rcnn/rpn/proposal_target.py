import mxnet as mx
import numpy as np
import numpy.random as npr

from rcnn.config import cfg
from helper.processing.bbox_regression import bbox_overlaps
from helper.processing.bbox_transform import bbox_transform


DEBUG = False

class ProposalTargetOperator(mx.operator.NumpyOp):
    def __init__(self, num_classes=21):
        super(ProposalTargetOperator, self).__init__(False)

        self._num_classes = num_classes
        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def list_arguments(self):
        return ['rpn_rois', 'gt_boxes', 'rpn_roi_pad', 'gt_pad']

    def list_outputs(self):
        return ['rois', 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights']

    def infer_shape(self, in_shape):
        batch = cfg.TRAIN.BATCH_SIZE
        roi_shape = (batch, 5)
        label_shape = (batch, )
        bbox_target_shape = (batch, self._num_classes * 4)
        bbox_inside_weights_shape = (batch, self._num_classes * 4)
        bbox_outside_weights_shape = (batch, self._num_classes * 4)
        return in_shape, [roi_shape, label_shape,
                bbox_target_shape, bbox_inside_weights_shape, bbox_outside_weights_shape]

    def forward(self, in_data, out_data):
        batch = cfg.TRAIN.BATCH_SIZE
        all_rois = in_data[0]
        gt_boxes = in_data[1]
        roi_pad = in_data[2][0]
        gt_pad = in_data[3][0]

        roi_size = int(all_rois.shape[0] - roi_pad)
        gt_size = int(gt_boxes.shape[0] - gt_pad)

        print "pt: roi_size: ", roi_size
        print "pt: gt_size: ", gt_size

        all_rois = all_rois[:roi_size, :]
        gt_boxes = gt_boxes[:gt_size, :]

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        #assert np.all(all_rois[:, 0] == 0), \
        #        'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        pad = batch - rois.shape[0]
        if pad > 0:
            print "Proposal Target Warning: No enough rois, pad %d sample" % pad
            bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        while pad > 0:
            rois = np.vstack((rois, rois[:pad, :]))
            labels = np.append(labels, labels[:pad])
            bbox_targets = np.vstack((bbox_targets, bbox_targets[:pad, :]))
            bbox_inside_weights = np.vstack((bbox_inside_weights,
                                             bbox_inside_weights[:pad, :]))
            bbox_outside_weights = np.vstack((bbox_outside_weights,
                                              bbox_outside_weights[:pad, :]))
            pad = batch - rois.shape[0]

        out_data[0][:] = rois
        out_data[1][:] = labels
        out_data[2][:] = bbox_targets
        out_data[3][:] = bbox_inside_weights
        out_data[4][:] = np.array(bbox_inside_weights > 0).astype(np.float32)

    def backward(self, out_grad, in_data, out_data, in_grad):
        pass

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
