"""
roidb
basic format [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
extended ['image', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
from helper.processing.bbox_process import unique_boxes, filter_small_boxes
from bbox_regression import compute_bbox_regression_targets
from rcnn.config import config


def prepare_roidb(image_set, num_classes):
    """
    add gt_overlaps, max_classes, max_overlaps to roidb
    :param image_set: sframe with ["image", "boxes", "gt_classes"]
    :return: sframe with ["image", "boxes", "gt_classes", "gt_overlaps",
                          "max_overlaps", "max_classes"]
    """
    tmp_overlaps = []
    tmp_max_overlaps = []
    tmp_max_classes = []
    print('prepare roidb')
    for item in image_set:
        gt_classes = item["gt_classes"]
        num_objs = len(gt_classes)
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
        for j in range(num_objs):
            overlaps[j, int(gt_classes[j])] = 1.0
        max_overlaps = overlaps.max(axis=1)
        max_classes = overlaps.argmax(axis=1)
        # background roi => background class
        zero_indexes = np.where(max_overlaps == 0)[0]
        assert(all(max_classes[zero_indexes] == 0))
        # foreground roi => foreground class
        nonzero_indexes = np.where(max_overlaps > 0)[0]
        assert(all(max_classes[nonzero_indexes] != 0))

        tmp_overlaps.append(list(overlaps))
        tmp_max_overlaps.append(max_overlaps)
        tmp_max_classes.append(max_classes)
    image_set["gt_overlaps"] = tmp_overlaps
    image_set["max_overlaps"] = tmp_max_overlaps
    image_set["max_classes"] = tmp_max_classes

def prepare_rpn_roidb(image_set):
    num_images = len(image_set)
    tmp_max_overlaps = []
    tmp_max_classes = []
    print("prepare_rpn_roidb")
    for i in range(num_images):
        gt_overlaps = np.vstack(image_set[i]["gt_overlaps"])
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        zero_indexes = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_indexes] == 0)
        nonzero_indexes = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_indexes] != 0)
        tmp_max_overlaps.append(max_overlaps)
        tmp_max_classes.append(max_classes)

    image_set["max_overlaps"] = tmp_max_overlaps
    image_set["max_classes"] = tmp_max_classes

def add_bbox_regression_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print('add bounding box regression targets')
    assert(len(roidb) > 0)
    assert('max_classes' in roidb[0])

    num_images = len(roidb)
    num_classes = len(roidb[0]['gt_overlaps'][0])
    tmp_bbox_target = []
    for im_i in range(num_images):
        rois = np.vstack(roidb[im_i]['boxes'])
        max_overlaps = np.asarray(roidb[im_i]['max_overlaps'])
        max_classes = np.asarray(roidb[im_i]['max_classes'])
        bbox_targets = compute_bbox_regression_targets(rois, max_overlaps, max_classes)
        tmp_bbox_target.append(bbox_targets)

    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (num_classes, 1))
        stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (num_classes, 1))
    else:
        # compute mean, std values
        class_counts = np.zeros((num_classes, 1)) + config.EPS
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in range(num_images):
            targets = tmp_bbox_target[im_i]
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # normalized targets

    for im_i in range(num_images):
        targets = tmp_bbox_target[im_i]
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] == cls)[0]
            tmp_bbox_target[im_i][cls_indexes, 1:] -= means[cls, :]
            tmp_bbox_target[im_i][cls_indexes, 1:] /= stds[cls, :]

    roidb["bbox_targets"] = [list(arr) for arr in tmp_bbox_target]
    return means.ravel(), stds.ravel()
