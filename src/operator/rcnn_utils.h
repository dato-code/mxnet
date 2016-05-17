/*!
 * Copyright (c) 2015 by Contributors
 * \file rcnn_utils.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RCNN_UTILS_H_
#define MXNET_OPERATOR_RCNN_UTILS_H_
#include <map>
#include <string>
#include <algorithm>
#include <mshadow/tensor.h>
#include <mshadow/extension.h>

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
  for (index_t i = 0; i < dets.shape_[0]; ++i) {
    i = static_cast<index_t>(order[i]);
    if (surprised[i] > 0.0f) {
      continue;
    }
    keep[(*out_size)++] = i;
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = area[i];
    for (index_t j = i + 1; j <dets.shape_[0]; ++j) {
      j = static_cast<index_t>(order[j]);
      if (surprised[j] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[j][0]);
      float yy1 = std::max(iy1, dets[j][1]);
      float xx2 = std::min(ix2, dets[j][2]);
      float yy2 = std::min(iy2, dets[j][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (iarea + area[j] - inter);
      if (ovr > thresh) {
        surprised[j] = 1.0f;
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

inline void _mk_anchors(const mshadow::Tensor<cpu, 1>& ws,
                const mshadow::Tensor<cpu, 1>& hs,
                float x_ctr,
                float y_ctr,
                size_t offset,
                mshadow::Tensor<cpu, 2> *out_anchors) {
  size_t num_anchors = ws.size(0);
  for (size_t i = 0; i < num_anchors; i++) {
    size_t index = i + offset*num_anchors;
    (*out_anchors)[index][0] = x_ctr - 0.5*(ws[i] - 1);
    (*out_anchors)[index][1] = y_ctr - 0.5*(hs[i] - 1);
    (*out_anchors)[index][2] = x_ctr + 0.5*(ws[i] - 1);
    (*out_anchors)[index][3] = y_ctr + 0.5*(hs[i] - 1);
  }
}


std::map<std::string, float> _find_whctrs(const mshadow::Tensor<cpu, 1>&
                                          anchor) {
  std::map<std::string, float> whctrs;
  whctrs["w"] = anchor[2] - anchor[0] + 1;
  whctrs["h"] = anchor[3] - anchor[1] + 1;
  whctrs["x_ctr"]  = anchor[0] + 0.5 * (whctrs["w"] -1);
  whctrs["y_ctr"] = anchor[1] + 0.5 * (whctrs["h"] - 1);
  return whctrs;
}

inline void _scale_enum(const mshadow::Tensor<cpu, 2>& ratio_anchors,
                const mshadow::Tensor<cpu, 1>& scales,
                mshadow::Tensor<cpu, 2> *scales_tempspace,
                mshadow::Tensor<cpu, 2> *out_anchors) {
  std::map<std::string, float> whctrs;
  mshadow::Tensor<cpu, 1> hs = (*scales_tempspace)[0];
  mshadow::Tensor<cpu, 1> ws = (*scales_tempspace)[1];

  for (size_t i = 0; i < ratio_anchors.size(0) ; i++) {
    whctrs = _find_whctrs(ratio_anchors[i]);
    ws = whctrs["w"] * scales;
    hs = whctrs["h"] * scales;

    size_t offset = i;
    _mk_anchors(ws, hs, whctrs["x_ctr"], whctrs["y_ctr"], offset, out_anchors);
  }
}


inline void _ratio_enum(const mshadow::Tensor<cpu, 1>& base_anchor,
                const mshadow::Tensor<cpu, 1>& ratios,
                mshadow::Tensor<cpu, 2> *ratios_tempspace,
                mshadow::Tensor<cpu, 2> *out_ratio_anchors ) {
  std::map<std::string, float> whctrs = _find_whctrs(base_anchor);
  float size = whctrs["w"] * whctrs["h"];
  mshadow::Tensor<cpu, 1> size_ratios = (*ratios_tempspace)[0];
  mshadow::Tensor<cpu, 1> ws = (*ratios_tempspace)[1];
  mshadow::Tensor<cpu, 1> hs = (*ratios_tempspace)[2];

  size_ratios = size/ratios;
  for (size_t i = 0; i < ratios.size(0); i++) {
    ws[i] = std::round(std::sqrt(size_ratios[i]));
    hs[i] = std::round(ws[i] * ratios[i]);
  }
  size_t offset = 0;
  _mk_anchors(ws, hs, whctrs["x_ctr"], whctrs["y_ctr"], offset,
              out_ratio_anchors);
}

inline void  generate_anchors(int base_size, const <cpu, 1>& ratios,
                       const mshadow::Tensor<cpu, 1>& scales,
                       mshadow::TensorContainer<cpu, 2>* out_anchors) {
  // Construct base anchor
  float base_anchor_data[] = {0, 0, base_size - 1, base_size -1};
  mshadow::TensorContainer<cpu, 1> base_anchor(Shape1(4));
  for (size_t i = 0; i < base_anchor.size(0); i++) {
    base_anchor[i] = base_anchor_data[i];
  }

  // enumerate aspect ratios
  mshadow::TensorContainer<cpu, 2> ratio_anchors(Shape2(ratios.size(0), 4));
  mshadow::TensorContainer<cpu, 2> ratios_tempspace(Shape2(3, ratios.size(0)));
  _ratio_enum(base_anchor, ratios, &ratios_tempspace, &ratio_anchors);

  // enumerate scales per aspect ratio
  mshadow::TensorContainer<cpu, 2> scales_tempspace(Shape2(2, scales.size(0)));
  (*out_anchors).Resize(Shape2(scales.size(0) * ratios.size(0), 4));
  _scale_enum(ratio_anchors, scales, &scales_tempspace, out_anchors);
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_RCNN_UTILS_H_
