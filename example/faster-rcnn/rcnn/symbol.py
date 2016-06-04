import mxnet as mx

from rpn import ProposalOperator
from rpn import AnchorTargetOperator
from rpn import ProposalTargetOperator

def get_symbol_vgg(num_classes=21, scales=[8, 16, 32]):
    """
    Faster R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    num_anchor = len(scales) * 3
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name='im_info')
    gt_boxes = mx.sym.Variable(name='gt_boxes')
    gt_pad = mx.sym.Variable(name='gt_pad')

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1",
        workspace=4096, attr={'lr_mult': '0', 'wd_mult':'0'})
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2",
        workspace=4096, attr={'lr_mult': '0', 'wd_mult':'0'})
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1",
        workspace=4096, attr={'lr_mult': '0', 'wd_mult':'0'})
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2",
        workspace=4096, attr={'lr_mult': '0', 'wd_mult':'0'})
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3",
        attr={'lr_mult': '0', 'wd_mult':'0'})
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    #======================== RPN
    rpn_conv = mx.sym.Convolution(name="rpn_conv_3x3", data=relu5_3, num_filter=512,
            pad=(1,1), kernel=(3,3), stride=(1,1), workspace=4096,)
            #attr={'lr_mult': '0', 'wd_mult':'0'})
    rpn_relu = mx.sym.Activation(name="rpn_relu", data=rpn_conv, act_type="relu")
    rpn_cls_score = mx.sym.Convolution(name="rpn_cls_score", data=rpn_relu,
            num_filter=num_anchor *2, pad=(0,0), kernel=(1,1), stride=(1,1))
            #attr={'lr_mult': '0', 'wd_mult':'0'})
    rpn_bbox_pred = mx.sym.Convolution(name="rpn_bbox_pred", data=rpn_relu,
            num_filter=4 * num_anchor, pad=(0,0), kernel=(1,1), stride=(1,1))
            #attr={'lr_mult': '0', 'wd_mult':'0'})
    rpn_cls_score_reshape = mx.sym.Reshape(name="rpn_cls_score_reshape",
            data=rpn_cls_score, shape=(0, 2, -1, 0))
    anchor_target_obj = AnchorTargetOperator(feat_stride=16, scales=scales)
    rpn_data = anchor_target_obj(name="rpn_data", rpn_cls_score=rpn_cls_score,
            gt_boxes=gt_boxes, im_info=im_info, gt_pad=gt_pad)
    rpn_labels = rpn_data[0]
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]
    # RPN loss
    rpn_loss_cls = mx.sym.SoftmaxOutput(name="rpn_loss_cls",
            data=rpn_cls_score_reshape, label=rpn_labels, multi_output=True, ignore_label=-1.,
            use_ignore=True, normalization="valid", grad_scale=1.0)
    rpn_bbox_loss = rpn_bbox_outside_weights * mx.sym.smooth_l1(data=rpn_bbox_inside_weights *
            (rpn_bbox_pred - rpn_bbox_targets), scalar=3.0)
    rpn_loss_bbox = mx.sym.MakeLoss(name='rpn_loss_bbox', data=rpn_bbox_loss, grad_scale=1.0)

    #======================= RoI Proposal
    rpn_cls_prob = mx.sym.SoftmaxActivation(name="rpn_cls_prob", data=rpn_cls_score_reshape,
            mode="channel")
    rpn_cls_prob_reshape = mx.sym.Reshape(name="rpn_cls_prob_reshape", data=rpn_cls_prob,
            shape=(0, 2 * num_anchor, -1, 0))
    proposal_obj = ProposalOperator(feat_stride=16, scales=[8,16,32])
    proposal = proposal_obj(name="rpn_rois", rpn_cls_prob=rpn_cls_prob_reshape,
            rpn_bbox_pred=rpn_bbox_pred, im_info=im_info)

    proposal_target_obj = ProposalTargetOperator(num_classes=num_classes)
    roi_data = proposal_target_obj(name="roi_data",
                                   rpn_rois=proposal[0], gt_boxes=gt_boxes,
                                   gt_pad=gt_pad, rpn_roi_pad=proposal[1])
    rois = roi_data[0]
    cls_labels = roi_data[1]
    bbox_targets = roi_data[2]
    bbox_inside_weights = roi_data[3]
    bbox_outside_weights = roi_data[4]
    #======================= RCNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=cls_labels,
            normalization="batch")
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_outside_weights * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=bbox_inside_weights * (bbox_pred - bbox_targets))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, normalization=True)
    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss,
                             rpn_loss_bbox, rpn_loss_cls,
                             cls_labels, bbox_targets])
    return group


def get_symbol_vgg_test(num_classes=21, scales=[4,8,16,32], get_feature=False):
    """
    Fast R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    num_anchor = len(scales) * 3
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    # RPN
    rpn_conv = mx.sym.Convolution(name="rpn_conv_3x3", data=relu5_3, num_filter=512,
            pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=2048)
    rpn_relu = mx.sym.Activation(name="rpn_relu", data=rpn_conv, act_type="relu")
    rpn_cls_score = mx.sym.Convolution(name="rpn_cls_score", data=rpn_relu,
            num_filter=2 * num_anchor,
            pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False, workspace=2048)
    rpn_bbox_pred = mx.sym.Convolution(name="rpn_bbox_pred", data=rpn_relu,
            num_filter=4 * num_anchor,
            pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False, workspace=2048)
    rpn_cls_score_reshape = mx.sym.Reshape(name="rpn_cls_score_reshape",
            data=rpn_cls_score, shape=(0, 2, -1, 0))

    # RoI Proposal
    rpn_cls_prob = mx.sym.SoftmaxActivation(name="rpn_cls_prob", data=rpn_cls_score_reshape, mode="channel")
    rpn_cls_prob_reshape = mx.sym.Reshape(name="rpn_cls_prob_reshape",
            data=rpn_cls_prob, shape=(0, 2 * num_anchor, -1, 0))
    proposal_obj = ProposalOperator(feat_stride=16, scales=scales, phase='TEST')
    proposal = proposal_obj(name="rpn_rois", rpn_cls_prob=rpn_cls_prob_reshape,
                             rpn_bbox_pred=rpn_bbox_pred, im_info=im_info)
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=proposal, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    # group output
    if get_feature == False:
        group = mx.symbol.Group([cls_prob, bbox_pred, proposal])
    else:
        fea_pool = mx.sym.Pooling(name="roi_avg_pool", data=pool5,
                kernel=(7,7), stride=(1,1), pool_type="avg")
        fea_flatten = mx.sym.Flatten(name="feature", data=fea_pool)
        group = mx.symbol.Group([cls_prob, bbox_pred, proposal, fea_flatten])
    return group
