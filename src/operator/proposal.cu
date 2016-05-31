/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./proposal-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ProposalParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
