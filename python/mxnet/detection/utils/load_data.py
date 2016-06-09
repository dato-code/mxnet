import numpy as np
from helper.processing.roidb import prepare_roidb, add_bbox_regression_targets
from helper.processing.roidb import prepare_rpn_roidb
from helper.processing.bbox_regression import bbox_overlaps

def load_gt_roidb(image_set, num_classes):
    prepare_roidb(image_set, num_classes)



def load_rpn_roidb(image_set, box_list, num_classes, is_train=False):
    print("load rpn")
    assert(len(image_set) == len(box_list))
    if is_train:
        # merge rpn_roidb into imageset
        num_image = len(image_set)
        out_boxes = []
        out_gt_classes = []
        out_gt_overlaps = []
        for i in range(num_image):
            boxes = np.vstack(box_list[i]["boxes"])
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, num_classes), dtype=np.float32)

            gt_boxes = np.vstack(image_set[i]["boxes"])
            gt_classes = np.asarray(image_set[i]["gt_classes"]).astype("int32")
            # n boxes and k gt_boxes => n * k overlap
            gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

            # for each box in n boxes, select only maximum overlap (must be greater than zero)
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0]
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]


            out_boxes.append(np.vstack((np.vstack(image_set[i]["boxes"]),
                                        boxes)))
            out_gt_classes.append(np.hstack((image_set[i]["gt_classes"],
                                             np.zeros((num_boxes,), dtype=np.float32))))
            out_gt_overlaps.append(np.vstack((np.vstack(image_set[i]["gt_overlaps"]),
                                              overlaps)))
        image_set["boxes"] = [list(x) for x in out_boxes]
        image_set["gt_classes"] = out_gt_classes
        image_set["gt_overlaps"] = [list(x) for x in out_gt_overlaps]
    else:
        # merge image into rpn_roidb
        image_set["boxes"] = box_list["boxes"]
    prepare_rpn_roidb(image_set)
    means, stds = add_bbox_regression_targets(image_set)
    print means, stds
    return image_set, means, stds


