/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal.cu
 * \brief proposal operator
*/
#include "./proposal-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ProposalParam param) {
  return new ProposalOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
