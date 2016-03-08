#include "../flexible_type.h"
#include <image/image_type.hpp>
#include <dmlc/logging.h>
#include <fstream>

using namespace graphlab;

void test_int() { 
  int y = 1;
  flexible_type x = y;
  CHECK_EQ((int)(x), y);
  CHECK_EQ(x.get_type(), flex_type_enum::INTEGER);
}

void test_float() { 
  float y = 3.0;
  flexible_type x = y;
  CHECK_EQ((float)(x), y);
  CHECK_EQ(x.get_type(), flex_type_enum::FLOAT);
}

void test_array() { 
  std::vector<double> y{1.0, 2.0, 3.0};
  flexible_type x = y;
  CHECK_EQ(x.get_type(), flex_type_enum::VECTOR);
  CHECK_EQ(x.size(), y.size());
  for (size_t i = 0; i < x.size(); ++i) {
    CHECK_EQ(x[i], y[i]);
  }
}

void test_image() {
  flex_image x;
  CHECK_EQ(x.is_decoded(), false);
  CHECK_EQ(x.m_height, 0);
  CHECK_EQ(x.m_width, 0);
  CHECK_EQ(x.m_channels, 0);
}

int main() {
  test_int();
  test_float();
  test_array();
  test_image();
  return 0;
}
