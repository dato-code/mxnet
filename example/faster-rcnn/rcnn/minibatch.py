"""
To construct data iterator from imdb, batch sampling procedure are defined here
training minibatch =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 3]}
    num_images should divide config['TRAIN_BATCH_SIZE'] and num_rois = config['TRAIN_BATCH_SIZE'] / num_images
validation minibatch is similar except num_images = 1 and num_rois = all rois
testing minibatch =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 3]}
    num_images = 1 and num_rois = all rois
"""
import cv2
import numpy as np
import numpy.random as npr

from helper.processing import image_processing
from rcnn.config import config


def get_minibatch(voc, idx, max_gt_boxes):
    """
    return minibatch of images in roidb
    :param roidb: subset of main database
    :param num_classes: number of classes is used in bbox regression targets
    :return: minibatch: {'data', 'rois', 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights'}
    """
    num_images = 1
    random_scale_indexes = npr.randint(0, high=len(config.TRAIN.SCALES), size=num_images)
    boxes = np.zeros((max_gt_boxes, 5), dtype="float32")
    pad = np.zeros((1,), dtype="float32")
    im_info = np.zeros((1, 3), dtype="float32")
    # im_array: [num_images, c, h, w]
    im_array, im_scales = get_image_array(voc, idx, config.TRAIN.SCALES, random_scale_indexes)
    # boxes, gt_classes
    gt_boxes = voc.load_pascal_annotation(idx)
    gt_boxes['boxes'][:] *= float(im_scales)
    bbox = np.hstack((gt_boxes['boxes'], gt_boxes['gt_classes']))
    boxes = np.zeros((max_gt_boxes, 5), dtype="float32")
    boxes[:bbox.shape[0], :] = bbox
    pad[0] = max_gt_boxes - bbox.shape[0]
    im_info[0, 0] = im_array.shape[2]
    im_info[0, 1] = im_array.shape[3]
    im_info[0, 2] = im_scales

    minibatch = {'data': im_array,
                 'im_info': im_info,
                 'gt_boxes': boxes,
                 'gt_pad': pad}
    return minibatch


def get_testbatch(voc, idx):
    """
    return test batch of given roidb
    actually, there is only one testing scale and len(roidb) is 1
    :param roidb: subset of main database
    :param num_classes: number of classes is used in bbox regression targets
    :return: minibatch: {'data', 'rois'}
    """
    num_images = 1
    random_scale_indexes = npr.randint(0, high=len(config.TRAIN.SCALES), size=num_images)
    im_info = np.zeros((1, 3), dtype="float32")
    im_array, im_scales = get_image_array(voc, idx, config.TRAIN.SCALES, random_scale_indexes)
    im_info[0] = im_array.shape[2]
    im_info[1] = im_array.shape[3]
    im_info[2] = im_scales

    testbatch = {'data': im_array,
                 'im_info': im_info}
    return testbatch


def get_image_array(voc, idx, scales, scale_indexes):
    """
    build image array from specific roidb
    :param roidb: images to be processed
    :param scales: scale list
    :param scale_indexes: indexes
    :return: array [b, c, h, w], list of scales
    """
    num_images = 1
    path = voc.image_path_from_index(idx)
    print path
    im = cv2.imread(path)
    # TODO(bing): support flip
    target_size = scales[scale_indexes[0]]
    im, im_scale = image_processing.resize(im, target_size, config.TRAIN.MAX_SIZE)
    im_tensor = image_processing.transform(im, config.PIXEL_MEANS)
    return im_tensor, im_scale


