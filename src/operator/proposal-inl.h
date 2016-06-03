/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal-inl.h
 * \brief
 * \author Piotr Teterwak
*/
#ifndef MXNET_OPERATOR_PROPOSAL_INL_H_
#define MXNET_OPERATOR_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./native_op-inl.h"
#include "./rcnn_utils.h"

namespace mxnet {
namespace op {

namespace proposal {
enum ProposalOpInputs {kClsProb, kBBoxPred, kImInfo};
enum ProposalOpOutputs {kOut, kTempProposal, kTempNMS};
enum ProposalOpResource {kTempSpace};
}  // proposal

struct AnchorInfo {
  std::vector<float> info;
};

inline std::istream &operator>>(std::istream &is, AnchorInfo &shape) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }

  float idx;
  std::vector<float> tmp;
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  shape.info = tmp;
  return is;
}


inline std::ostream &operator<<(std::ostream &os, const AnchorInfo &shape) {
  os << '(';
  for (index_t i = 0; i < shape.info.size(); ++i) {
    if (i != 0) os << ',';
    os << shape.info[i];
  }
  // python style tuple
  if (shape.info.size() == 1) os << ',';
  os << ')';
  return os;
}

struct ProposalParam : public dmlc::Parameter<ProposalParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  AnchorInfo scales;
  AnchorInfo ratios;
  AnchorInfo base_anchor;
  int feature_stride;
  DMLC_DECLARE_PARAMETER(ProposalParam) {
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    DMLC_DECLARE_FIELD(scales).set_default(AnchorInfo())
    .describe("Used to generate anchor windows by enumerating scales");
    DMLC_DECLARE_FIELD(ratios).set_default(AnchorInfo())
    .describe("Used to generate anchor windows by enumerating ratios");
    DMLC_DECLARE_FIELD(base_anchor).set_default(AnchorInfo())
    .describe("The base anchor that is used as reference anchor for generating anchors.");
    DMLC_DECLARE_FIELD(feature_stride).set_default(1)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
  }
};

template<typename xpu>
class ProposalOp : public NativeOpBase<xpu> {
 public:
  explicit ProposalOp(ProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Parent::_InitForward(ctx, in_data, out_data, aux_states);
    Parent::_SyncData(in_data, Parent::in_data_ptr_, s, nativeop::kTensorToData);
    if (s != NULL) s->Wait();
 
    size_t num_anchors = in_data[proposal::kClsProb].shape_[1] / 2;
    Shape<4> scores_shape = Shape4(in_data[proposal::kClsProb].shape_[0],
                            in_data[proposal::kClsProb].shape_[1] / 2,
                            in_data[proposal::kClsProb].shape_[2],
                            in_data[proposal::kClsProb].shape_[3]);
  

    Shape<4> bbox_deltas_shape = Shape4(in_data[proposal::kBBoxPred].shape_[0],
                            in_data[proposal::kBBoxPred].shape_[1],
                            in_data[proposal::kBBoxPred].shape_[2],
                            in_data[proposal::kBBoxPred].shape_[3]);

    Shape<2> im_info_shape = Shape2(in_data[proposal::kImInfo].shape_[0],
                            in_data[proposal::kImInfo].shape_[1]);

    Shape<3> out_shape = Shape3(out_data[proposal::kOut].shape_[0],
                            out_data[proposal::kOut].shape_[1],
                            out_data[proposal::kOut].shape_[2]);

    Shape<2> workspace_proposals_shape = Shape2(out_data[proposal::kTempProposal].shape_[0],
                                                out_data[proposal::kTempProposal].shape_[1]);

    Shape<2> workspace_nms_shape = Shape2(out_data[proposal::kTempNMS].shape_[0],
                                          out_data[proposal::kTempNMS].shape_[1]);

    real_t* foreground_score_pointer =
      Parent::in_data_ptr_[proposal::kClsProb] + scores_shape.Size();

    Tensor<cpu, 4> scores =  Tensor<cpu, 4>(foreground_score_pointer, scores_shape);
    Tensor<cpu, 4> bbox_deltas = Tensor<cpu, 4>(Parent::in_data_ptr_[proposal::kBBoxPred],
                                                bbox_deltas_shape);
    Tensor<cpu, 2> im_info = Tensor<cpu, 2>(Parent::in_data_ptr_[proposal::kImInfo],
                                            im_info_shape);

    Tensor<cpu, 3> out = Tensor<cpu, 3>(Parent::out_data_ptr_[proposal::kOut],
                                        out_shape);

    Tensor<cpu, 2> workspace_proposals = Tensor<cpu, 2>(Parent::out_data_ptr_[proposal::kTempProposal],
                                                        workspace_proposals_shape);

    Tensor<cpu, 2> workspace_nms = Tensor<cpu, 2>(Parent::out_data_ptr_[proposal::kTempNMS],
                                                  workspace_nms_shape);

    index_t height = scores.size(2);
    index_t width = scores.size(3);


    utils::GenerateAnchors(param_.base_anchor.info,
                    param_.ratios.info,
                    param_.scales.info,
                    &workspace_proposals);

    //Enumerate all shifted anchors
    for (index_t i = 0; i < num_anchors; ++i){
      for (index_t j = 0; j < height; ++j){
        for (index_t k = 0; k < width; ++k){
          index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
          workspace_proposals[index][0] = workspace_proposals[i][0] + k * param_.feature_stride;
          workspace_proposals[index][1] = workspace_proposals[i][1] + j * param_.feature_stride;
          workspace_proposals[index][2] = workspace_proposals[i][2] + k * param_.feature_stride;
          workspace_proposals[index][3] = workspace_proposals[i][3] + j * param_.feature_stride;
        }
      }
    }

    //Copy scores to workspace.
    for (index_t i = 0; i < num_anchors; i++) {
      for (index_t h = 0; h < height; h++) {
        for (index_t w = 0; w < width; w++) {
          index_t index = h * (width * num_anchors) + w * (num_anchors) + i;
          workspace_proposals[index][4] = scores[0][i][h][w];
        }
      }
    }

    utils::BBoxTransformInv(workspace_proposals, bbox_deltas, &(workspace_proposals));
    utils::ClipBoxes(Shape2(im_info[0][0],im_info[0][1]), &(workspace_proposals));

    float scale = im_info[0][2];
    index_t out_size;
    Tensor<cpu, 1> output = workspace_nms[4];

    utils::NonMaximumSuppression(workspace_proposals,
                          param_.threshold,
                          param_.rpn_min_size * scale, param_.rpn_pre_nms_top_n,
                          param_.rpn_post_nms_top_n,
                          &workspace_nms,
                          &output,
                          &out_size);

    for (index_t i = 0; i < out_size; ++i) {
      index_t index = output[i];
      //batch index 0
      out[i][0] = 0;
      for (index_t j = 1; j < 5; ++j) {
        out[i][j] = workspace_proposals[index][j];
      }
    }
    Parent::_SyncData(out_data, Parent::out_data_ptr_, s, nativeop::kDataToTensor);
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

 private:
  ProposalParam param_;
  typedef NativeOpBase<xpu> Parent;
};  // class ProposalOp

template<typename xpu>
Operator *CreateOp(ProposalParam param);


#if DMLC_USE_CXX11
class ProposalProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[cls_prob, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(proposal::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape;
    bbox_pred_shape = Shape4(dshape[0], dshape[1] * 2, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kBBoxPred,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(1,3);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kImInfo, im_info_shape);
    out_shape->clear();
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 5));
    out_shape->push_back(Shape2(dshape[1] / 2 * dshape[2] * dshape[3], 5));
    out_shape->push_back(Shape2(5, dshape[1] / 2 * dshape[2] * dshape[3]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProposalProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Proposal";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  //todo fix all parameters
  std::vector<std::string> ListArguments() const override {
    return {"rpn_cls_score", "rpn_bbox_pred", "im_info"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"proposals", "temp_proposal", "temp_nms" };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                             std::vector<int>* in_type) const override;


 private:
  ProposalParam param_;
};  // class ProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  //  MXNET_OPERATOR_PROPOSAL_INL_H_

