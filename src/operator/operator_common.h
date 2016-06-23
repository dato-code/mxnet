/*!
 * Copyright (c) 2015 by Contributors
 * \file  operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_

#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace mxnet {
namespace op {

/*!
* \brief structure for numerical tuple input
* \tparam VType data type of param
*/
template<typename VType>
struct NumericalParam {
  NumericalParam() {}
  explicit NumericalParam(VType *begin, VType *end) {
    int32_t size = static_cast<int32_t>(end - begin);
    info.resize(size);
    for (int i = 0; i < size; ++i) {
      info[i] = *(begin + i);
    }
  }
  inline size_t ndim() const {
    return info.size();
  }
  std::vector<VType> info;
};

template<typename VType>
inline std::istream &operator>>(std::istream &is, NumericalParam<VType> &param) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  VType idx;
  std::vector<VType> tmp;
  // deal with empty case
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    param.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
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
  param.info = tmp;
  return is;
}

template<typename VType>
inline std::ostream &operator<<(std::ostream &os, const NumericalParam<VType> &param) {
  os << '(';
  for (index_t i = 0; i < param.info.size(); ++i) {
    if (i != 0) os << ',';
    os << param.info[i];
  }
  // python style tuple
  if (param.info.size() == 1) os << ',';
  os << ')';
  return os;
}

/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define Assign(out, req, exp)           \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }


/*! \brief exception throwed by InferShape error */
struct InferShapeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferShapeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*! \brief exception throwed by InferShape error */
struct InferTypeError {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferTypeError(std::string msg, int index)
    : msg(msg), index(index) {}
};

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param shape_array the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                   \
  {                                                                     \
    auto &out = (shape_array)[index];                                   \
    if (out.ndim() == 0) {                                              \
      out = shape;                                                      \
    } else {                                                            \
      if (out != shape) {                                               \
        std::ostringstream os;                                          \
        os << "Shape inconsistent, Provided " <<  '='<< out << ','      \
           << " inferred shape=" << shape;                              \
        throw ::mxnet::op::InferShapeError(os.str(), index);            \
      }                                                                 \
    }                                                                   \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define TYPE_ASSIGN_CHECK(type_array, index, type)                      \
  {                                                                     \
    auto &out = (type_array)[index];                                    \
    if (out == -1) {                                                    \
      out = type;                                                       \
    } else {                                                            \
      if (out != type) {                                                \
        std::ostringstream os;                                          \
        os << "Type inconsistent, Provided " <<  '='<< out << ','       \
           << " inferred type=" << type;                                \
        throw ::mxnet::op::InferTypeError(os.str(), index);             \
      }                                                                 \
    }                                                                   \
  }

// helper macro to implement bind dispatch
#if MXNET_USE_CUDA
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_OPERATOR_COMMON_H_
