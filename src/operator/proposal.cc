/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal.cc
 * \brief
 * \author Piotr Teterwak
*/

#include "./proposal-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ProposalParam param) {
  return new ProposalOp<cpu>(param);
}

Operator *ProposalProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> * in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ProposalParam);

MXNET_REGISTER_OP_PROPERTY(Proposal, ProposalProp)
.describe("Generate region proposals via RPN")
.add_argument("rpn_cls_score", "Symbol", "Score of how likely proposal is object.")
.add_argument("rpn_bbox_pred", "Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "Symbol", "Image size and scale.")
.add_arguments(ProposalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

