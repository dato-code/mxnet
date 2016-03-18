#include<flexible_type/flexible_type.hpp>

extern "C" {

bool sa_callback(const graphlab::flexible_type* value) {
  return true;
}

bool sf_callback(const graphlab::flexible_type** value, size_t row) {
  return true;
}

}
