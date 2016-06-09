import argparse
import logging
import os

import mxnet as mx

try:
    import graphlab as gl
except:
    import sframe as gl

from rcnn.config import config
from rcnn.data_iter import AnchorLoader, ROIIter
from rcnn.solver import Solver
from rcnn.symbol import get_vgg_rpn, get_vgg_rpn_test, get_vgg_rcnn
from utils.load_data import load_gt_roidb, load_rpn_roidb
from utils.load_model import load_checkpoint, load_param
from utils.save_model import save_checkpoint


def train_rpn(image_set, num_classes, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent):
    # load symbol
    sym = get_vgg_rpn()
    feat_sym = get_vgg_rpn().get_internals()['rpn_cls_score_output']

    # load training data
    #voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path, flip=True)
    load_gt_roidb(image_set, num_classes)
    train_data = AnchorLoader(feat_sym, image_set, batch_size=1, shuffle=False, mode='train')

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx)

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['rpn_conv_3x3_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'], ctx=ctx)
    args['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'], ctx=ctx)
    args['rpn_cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_cls_score_weight'], ctx=ctx)
    args['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'], ctx=ctx)
    args['rpn_bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'], ctx=ctx)
    args['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'], ctx=ctx)

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=1e-3, lr_scheduler=mx.lr_scheduler.FactorScheduler(60000, 0.1))
    solver.fit(train_data, frequent=frequent)


def test_rpn(image_set, num_classes, trained, epoch, ctx, rpn_outpath):
    from rcnn.rpn.generate import Detector, generate_detections

    # load symbol
    sym = get_vgg_rpn_test()

    # load testing data
    #voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path)
    #test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')
    load_gt_roidb(image_set, num_classes)
    test_data = ROIIter(image_set, batch_size=1, shuffle=False, mode='test')

    # load trained
    args, auxs = load_param(trained, epoch, convert=True, ctx=ctx)

    # start testing
    detector = Detector(sym, ctx, args, auxs)
    imdb_boxes = generate_detections(detector, test_data, rpn_outpath, vis=False)
    #voc.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def train_rcnn(image_set, num_classes, rpn_sfame, pretrained, epoch,
               prefix, ctx, begin_epoch, end_epoch, frequent):
    # load symbol
    sym = get_vgg_rcnn()

    # load training data
    #voc, roidb, means, stds = load_rpn_roidb(image_set, year, root_path, devkit_path, flip=True)
    #train_data = ROIIter(roidb, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train')
    load_gt_roidb(image_set, num_classes)
    load_rpn_roidb(image_set, rpn_sframe, num_classes, True)
    train_data = ROIIter(image_set, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train')

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx)

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['cls_score_weight'], ctx=ctx)
    args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'], ctx=ctx)
    args['bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.001, shape=arg_shape_dict['bbox_pred_weight'], ctx=ctx)
    args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'], ctx=ctx)

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=1e-3, lr_scheduler=mx.lr_scheduler.FactorScheduler(30000, 0.1))
    solver.fit(train_data, frequent=frequent)

    # edit params and save
    for epoch in range(begin_epoch + 1, end_epoch + 1):
        arg_params, aux_params = load_checkpoint(prefix, epoch)
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds, ctx=ctx)).T
        arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds, ctx=ctx) + \
                                       mx.nd.array(means, ctx=ctx)
        save_checkpoint(prefix, epoch, arg_params, aux_params)


def alternate_train(train_path, num_classes, pretrained, epoch,
                    ctx, begin_epoch, rpn_epoch, rcnn_epoch, rpn_out_path, frequent):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info('########## TRAIN RPN WITH IMAGENET INIT')
    #image_set = gl.SFrame(train_path)
    #config.TRAIN.HAS_RPN = True
    #config.TRAIN.BATCH_SIZE = 1
    #train_rpn(image_set, num_classes, pretrained, epoch,
    #          'model/rpn1', ctx, begin_epoch, rpn_epoch, frequent)

    logging.info('########## GENERATE RPN DETECTION')
    image_set = gl.SFrame(train_path)
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(image_set, num_classes, 'model/rpn1', rpn_epoch, ctx, rpn_out_path)

    logging.info('########## TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    image_set = gl.SFrame(train_path)
    rpn_data = gl.SFrame(rpn_outpath)
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    train_rcnn(image_set, num_classes, pretrained, rpn_data, epoch,
               'model/rcnn1', ctx, begin_epoch, rcnn_epoch, frequent)

    logging.info('########## TRAIN RPN WITH RCNN INIT')
    image_set = gl.SFrame(train_path)
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    config.TRAIN.FINETUNE = True
    train_rpn(image_set, num_classes, 'model/rcnn1', rcnn_epoch,
              'model/rpn2', ctx, begin_epoch, rpn_epoch, frequent)

    logging.info('########## GENERATE RPN DETECTION')
    image_set = gl.SFrame(train_path)
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(image_set, num_classes, 'model/rpn2', rpn_epoch, ctx, rpn_out_path)

    logger.info('########## TRAIN RCNN WITH RPN INIT AND DETECTION')
    image_set = gl.SFrame(train_path)
    rpn_data = gl.SFrame(rpn_outpath)
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    train_rcnn(image_set, num_classes, rpn_data, 'model/rpn2', rpn_epoch,
               'model/rcnn2', ctx, begin_epoch, rcnn_epoch, frequent)

