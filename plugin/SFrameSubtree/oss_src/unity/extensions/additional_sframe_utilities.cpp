/**
 * Copyright (C) 2015 Dato, Inc.
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
#include <string>
#include <vector>
#include <unity/lib/gl_sarray.hpp>
#include <unity/lib/gl_sframe.hpp>
#include <unity/lib/toolkit_function_macros.hpp>
using namespace graphlab;

void sarray_callback(graphlab::gl_sarray input, size_t callback_addr, size_t callback_data_addr,
                     size_t begin, size_t end, size_t bias) {
  typedef int(*callback_type)(void*, const flexible_type*, size_t, size_t, size_t);
  callback_type callback_fun = (callback_type)(callback_addr);
  auto ra = input.range_iterator(begin, end);
  auto iter = ra.begin();
  size_t idx = bias;
  size_t sz = end - begin;
  while (iter != ra.end()) {
    int unsuccess = callback_fun((void*)(callback_data_addr), &(*iter), 1, idx++, sz);
    if (unsuccess) log_and_throw("Error applying callback");
    ++iter;
  }
}

void sframe_callback(graphlab::gl_sframe input, size_t callback_addr, size_t callback_data_addr,
                     size_t begin, size_t end, size_t bias) {
  ASSERT_MSG(input.num_columns() > 0, "SFrame has no column");
  typedef int(*callback_type)(void*, const flexible_type*, size_t, size_t, size_t);
  callback_type callback_fun = (callback_type)(callback_addr);
  auto ra = input.range_iterator(begin, end);
  std::vector<flexible_type> row_vec;
  auto iter = ra.begin();
  size_t idx = bias;
  size_t sz = end - begin;
  while (iter != ra.end()) {
    row_vec = (*iter);
    int unsuccess = callback_fun((void*)(callback_data_addr), &(row_vec[0]),
                                row_vec.size(), idx++, sz);

    if (unsuccess) log_and_throw("Error applying callback");
    ++iter;
  }
}

BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(sarray_callback, "input", "callback_addr", "callback_data", "begin", "end", "bias");
REGISTER_FUNCTION(sframe_callback, "input", "callback_addr", "callback_data", "begin", "end", "bias");
END_FUNCTION_REGISTRATION
