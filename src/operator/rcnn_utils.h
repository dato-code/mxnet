/*!
 * Copyright (c) 2015 by Contributors
 * \file rcnn_utils.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RCNN_UTILS_H_
#define MXNET_OPERATOR_RCNN_UTILS_H_
#include <algorithm>
#include <mshadow/tensor.h>
#include <mshadow/extension.h>

namespace mshadow {
namespace expr {
//=========================
// BBox Overlap mshadow exp
//=========================
/*!
 * \brief 2D overlap expression
 * \tparam SrcExp source expression to be calculated
 * \tparam DType data type
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct OverlapExp: public MakeTensorExp<OverlapExp<SrcExp, DType, srcdim>,
                                        SrcExp, srcdim, DType> {
  /*! \brief boxes */
  const SrcExp &lhs_;
  /*! \brief query_boxes */
  const SrcExp &rhs_;
  /*! \brief constructor */
  explicit OverlapExp(const SrcExp &lhs, const SrcExp &rhs)
    : lhs_(lhs), rhs_(rhs) {
    // lhs shape: (N, 4)
    // rhs shape: (K, 4)
    // output : (N, K)
    CHECK_EQ(srcdim, 2) << "Input must be 2D Tensor";
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(lhs_);
    Shape<2> rhs_shape = ShapeCheck<srcdim, SrcExp>::Check(rhs_);
    CHECK_EQ(this->shape_[1], 4) << "boxes must be in shape (N, 4)";
    CHECK_EQ(rhs_shape[1], 4) << "query box must be in shape (K, 4)";
    this->shape_[1] = rhs_shape[0];
  }
};  // struct OverlapExp

/*!
 * \brief calcuate overlaps between boxes and query boxes
 * \param lhs boxes
 * \param rhs query boxes
 * \tparam SrcExp source expression
 * \tparam DType data type
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int etype>
inline OverlapExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
bbox_overlaps(const Exp<SrcExp, DType, etype> &lhs,
             const Exp<SrcExp, DType, etype> &rhs) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return OverlapExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
    (lhs.self(), rhs.self());
}

//----------------------
// Execution plan
//----------------------

template<typename SrcExp, typename DType, int srcdim>
struct Plan<OverlapExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const OverlapExp<SrcExp, DType, srcdim> &e)
    : lhs_(MakePlan(e.lhs_)),
      rhs_(MakePlan(e.rhs_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    DType box_area =
      (rhs_.Eval(j, 2) - rhs_.Eval(j, 0) + 1) *
      (rhs_.Eval(j, 3) - rhs_.Eval(j, 1) + 1);
    DType iw =
      (lhs_.Eval(i, 2) < rhs_.Eval(j, 2) ? lhs_.Eval(i, 2) : rhs_.Eval(j, 2)) -
      (lhs_.Eval(i, 0) > rhs_.Eval(j, 0) ? lhs_.Eval(i, 0) : rhs_.Eval(j, 0)) + 1;
    if (iw < 0.0f) {
      return DType(0.0f);
    } else {
      DType ih =
        (lhs_.Eval(i, 3) < rhs_.Eval(j, 3) ? lhs_.Eval(i, 3) : rhs_.Eval(j, 3)) -
        (lhs_.Eval(i, 1) > rhs_.Eval(j, 1) ? lhs_.Eval(i, 1) : rhs_.Eval(j, 1)) + 1;
      if (ih < 0.0f) {
        return DType(0.0f);
      } else {
        DType ua =
          (lhs_.Eval(i, 2) - lhs_.Eval(i, 0) + 1) *
          (lhs_.Eval(i, 3) - lhs_.Eval(i, 1) + 1) +
          box_area - iw * ih;
        return DType(iw * ih / ua);
      }
    }
  }

 private:
  Plan<SrcExp, DType> lhs_;
  Plan<SrcExp, DType> rhs_;
};  // struct Plan

}  // namespace expr
}  // namespace mshadow

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace utils {

struct ReverseArgsortCompl {
  const float *val_;
  explicit ReverseArgsortCompl(float *val)
    : val_(val) {}
  bool operator() (float i, float j) {
    return (val_[static_cast<index_t>(i)] >
            val_[static_cast<index_t>(j)]);
  }
};

inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2> &dets,
                                  const float thresh,
                                  const float min_size,
                                  const index_t pre_nms_top_n,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 2> *tempspace,
                                  mshadow::Tensor<cpu, 1> *output,
                                  index_t *out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_EQ(dets.shape_[0], tempspace->shape_[1]);
  CHECK_EQ(tempspace->shape_[0], 4);
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(tempspace->CheckContiguous(), true);
  mshadow::Tensor<cpu, 1> score = (*tempspace)[0];
  mshadow::Tensor<cpu, 1> area = (*tempspace)[1];
  mshadow::Tensor<cpu, 1> order = (*tempspace)[2];
  mshadow::Tensor<cpu, 1> surprised = (*tempspace)[3];
  mshadow::Tensor<cpu, 1> keep = *output;
  // copy score, calculate area, init order
  for (index_t i = 0; i < dets.size(0); ++i) {
    area[i] = (dets[i][2] - dets[i][0] + 1) *
              (dets[i][3] - dets[i][1] + 1);
    score[i] = dets[i][4];
    order[i] = i;
  }

  // argsort to get order
  ReverseArgsortCompl cmpl(score.dptr_);
  std::sort(order.dptr_, order.dptr_ + dets.size(0), cmpl);

  // calculate nms
  *out_size = 0;
  index_t valid_size= 0;
  for (index_t k = 0; k < dets.size(0) && valid_size <
       pre_nms_top_n && (*out_size) < post_nms_top_n; ++k) {
    index_t i = static_cast<index_t>(order[k]);
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = area[i];
    float ih = ix2 - ix1 + 1;
    float iw = iy2 - iy1 + 1;

    if (ih < min_size || iw < min_size) {
      valid_size++;
      continue;
    }
    if (surprised[i] > 0.0f ) {
      continue;
    }

    keep[(*out_size)++] = i;
    for (index_t j = k + 1; j <dets.shape_[0]; ++j) {
      index_t l = static_cast<index_t>(order[j]);
      if (surprised[l] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[l][0]);
      float yy1 = std::max(iy1, dets[l][1]);
      float xx2 = std::min(ix2, dets[l][2]);
      float yy2 = std::min(iy2, dets[l][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (iarea + area[l] - inter);
      if (ovr > thresh ) {
        surprised[l] = 1.0f;
      }
    }
  }
}



}  // namespace utils
}  // namespace op
}  // namespace mxnet

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace utils {

inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        mshadow::Tensor<cpu, 1> *out_anchors) {
  (*out_anchors)[0] = x_ctr - 0.5*(w - 1);
  (*out_anchors)[1] = y_ctr - 0.5*(h - 1);
  (*out_anchors)[2] = x_ctr + 0.5*(w - 1);
  (*out_anchors)[3] = y_ctr + 0.5*(h - 1);
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       mshadow::Tensor<cpu, 1> *out_anchor) {
  float w = base_anchor[2] - base_anchor[1] + 1;
  float h = base_anchor[3] - base_anchor[1] + 1;
  float x_ctr = base_anchor[0] + 0.5 * (w-1);
  float y_ctr = base_anchor[1] + 0.5 * (h-1);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = int(std::sqrt(size_ratios) + 0.5) * scale;
  float new_h = int(new_w / scale * ratio + 0.5) * scale;

  _MakeAnchor(new_w, new_h, x_ctr,
             y_ctr, out_anchor);
}

// out_anchors must have shape (n,4), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const std::vector<float>& ratios,
                            const std::vector<float>& scales,
                            mshadow::Tensor<cpu, 2>* out_anchors) {
  CHECK_GE(out_anchors->size(0), ratios.size() * scales.size());
  CHECK_EQ(out_anchors->size(1), 4);
  size_t i = 0;
  for (size_t j = 0; j < ratios.size(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      mshadow::Tensor<cpu, 1> out_anchor = (*out_anchors)[i];
      _Transform(scales[k], ratios[j], base_anchor, &out_anchor);
      ++i;
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

inline void BBoxTransform(const mshadow::Tensor<cpu, 2>& ex_rois,
                   const mshadow::Tensor<cpu, 2>& gt_rois,
                   mshadow::Tensor<cpu, 2> *out_targets) {
  CHECK_EQ(ex_rois.size(1), 4);
  CHECK_EQ(gt_rois.size(1), 4);
  CHECK_EQ(out_targets->size(1), 4);
  CHECK_EQ(ex_rois.size(0), gt_rois.size(0));
  CHECK_EQ(gt_rois.size(0), out_targets->size(0));

  size_t num_rois = ex_rois.size(0);

  for (size_t i=0; i < num_rois; ++i) {
    float ex_width = ex_rois[i][2] - ex_rois[i][0] + 1.0;
    float ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.0;
    float ex_ctr_x = ex_rois[i][0] + 0.5 * ex_width;
    float ex_ctr_y = ex_rois[i][1] + 0.5 * ex_height;

    float gt_width = gt_rois[i][2] - gt_rois[i][0] + 1.0;
    float gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.0;
    float gt_ctr_x = gt_rois[i][0] + 0.5 * gt_width;
    float gt_ctr_y = gt_rois[i][1] + 0.5 * gt_height;

    float target_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
    float target_dy = (gt_ctr_y - ex_ctr_y)/ ex_height;
    float target_dw = log(gt_width) - log(ex_width);
    float target_dh = log(gt_height) - log(ex_height);

    (*out_targets)[i][0] = target_dx;
    (*out_targets)[i][1] = target_dy;
    (*out_targets)[i][2] = target_dw;
    (*out_targets)[i][3] = target_dh;
  }
}

inline void BBoxTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                      const mshadow::Tensor<cpu, 4>& deltas,
                      mshadow::Tensor<cpu, 2> *out_pred_boxes) {
  CHECK_EQ(boxes.size(1), 4);
  CHECK_EQ(out_pred_boxes->size(1), 4);
  size_t anchors = deltas.size(1)/4;
  size_t heights = deltas.size(2);
  size_t widths = deltas.size(3);

  for (size_t a = 0; a < anchors; ++a) {
    for (size_t h = 0; h < heights; ++h) {
      for (size_t w = 0; w < widths; ++w) {
        index_t index = h * (widths * anchors) + w * (anchors) + a;
        float width = boxes[index][2] - boxes[index][0] + 1.0;
        float height = boxes[index][3] - boxes[index][1] + 1.0;
        float ctr_x = boxes[index][0] + 0.5 * width;
        float ctr_y = boxes[index][1] + 0.5 * height;

        float dx = deltas[0][a*4 + 0][h][w];
        float dy = deltas[0][a*4 + 1][h][w];
        float dw = deltas[0][a*4 + 2][h][w];
        float dh = deltas[0][a*4 + 3][h][w];

        float pred_ctr_x = dx * width + ctr_x;
        float pred_ctr_y = dy * height + ctr_y;
        float pred_w = exp(dw) * width;
        float pred_h = exp(dh) * height;

        (*out_pred_boxes)[index][0] = pred_ctr_x - 0.5 * pred_w;
        (*out_pred_boxes)[index][1] = pred_ctr_y - 0.5 * pred_h;
        (*out_pred_boxes)[index][2] = pred_ctr_x + 0.5 * pred_w;
        (*out_pred_boxes)[index][3] = pred_ctr_y + 0.5 * pred_h;
      }
    }
  }
}

inline void ClipBoxes(const mshadow::Shape<2>& im_shape, mshadow::Tensor<cpu, 2> *in_out_boxes) {
  CHECK_EQ(in_out_boxes->size(1), 4);
  size_t num_boxes = in_out_boxes->size(0);

  for (size_t i=0; i < num_boxes; ++i) {
    (*in_out_boxes)[i][0] = std::max(std::min((*in_out_boxes)[i][0],
          static_cast<float> (im_shape[1] - 1)), static_cast<float>(0));
    (*in_out_boxes)[i][1] = std::max(std::min((*in_out_boxes)[i][1],
          static_cast<float> (im_shape[0] - 1)), static_cast<float>(0));
    (*in_out_boxes)[i][2] = std::max(std::min((*in_out_boxes)[i][2],
          static_cast <float>(im_shape[1] - 1)), static_cast<float>(0));
    (*in_out_boxes)[i][3] = std::max(std::min((*in_out_boxes)[i][3],
          static_cast <float>(im_shape[0] - 1)), static_cast<float>(0));
  }
}



}  // namespace utils
}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_RCNN_UTILS_H_
