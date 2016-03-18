/*!
 * Copyright (c) 2015 by Contributors
 * \file flexible_type.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_FLEXIBLE_TYPE_H_
#define MXNET_FLEXIBLE_TYPE_H_

#define _ASSERTIONS_H_
#define GRAPHLAB_LOG_LOG_HPP
// -----------------------
#define ASSERT_NE CHECK_NE
#define ASSERT_GE CHECK_GE
#define ASSERT_GT CHECK_GT
#define ASSERT_LE CHECK_LE
#define ASSERT_LT CHECK_LT
#define ASSERT_EQ CHECK_EQ
#define ASSERT_MSG(FALSE, MSG, ...) \
  LOG(FATAL) << MSG;
#define ASSERT_FALSE(VAR) \
  CHECK_EQ(VAR, false);
#define DASSERT_FALSE(cond) ASSERT_FALSE(cond)
#define log_and_throw(msg) \
  LOG(FATAL) << msg;
#include<flexible_type/flexible_type.hpp>


#endif  // MXNET_FLEXIBLE_TYPE_H_
