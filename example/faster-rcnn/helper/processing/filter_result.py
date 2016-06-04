import nms
from bbox_transform import bbox_transform_inv, clip_boxes


def filter_result(num_classes, scores, box_deltas, rois, thresh=0.05):
    result = {}
    boxes = rois[:, 1:5] / im_info[0][2]
    boxes = bbox_transform_inv(boxes, box_deltas)
    boxes = clip_boxes(boxes, im.shape)
    for j in range(num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=True)
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        cls_scores = scores[keep, j]
        result[j] = {}
        result[j]["score"] = cls_scores
        result[j]["dets"] = cls_dets


