// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef MICRO_TEST_CCUTILS_MICRO_OPS_GTEST_UTILS_H_
#define MICRO_TEST_CCUTILS_MICRO_OPS_GTEST_UTILS_H_

#include "gtest/gtest.h"
#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/include/public/micro.h"
#include "micro/include/utils/macros.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

template<typename T>
inline void ExpectEqual(const T &a, const T &b) {
  EXPECT_EQ(a, b);
}

template<>
inline void ExpectEqual<float>(const float &a, const float &b) {
  EXPECT_FLOAT_EQ(a, b);
}

template<>
inline void ExpectEqual<double>(const double &a, const double &b) {
  EXPECT_DOUBLE_EQ(a, b);
}

template<typename EXP_TYPE,
    typename RES_TYPE,
    bool is_fp = true>
struct Expector;

// Partial specialization for float and double.
template<typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, true> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(
      const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
      const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size) {
    AssertSameDims(x_dims, x_dim_size, y_dims, y_dim_size);
    const int32_t size = base::GetShapeSize(x_dim_size, x_dims);
    for (int32_t i = 0; i < size; ++i) {
      ExpectEqual(x[i], y[i]);
    }
  }

  static void Near(
      const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
      const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size,
      const double rel_err, const double abs_err) {
    AssertSameDims(x_dims, x_dim_size, y_dims, y_dim_size);
    if (x_dim_size == 4) {
      for (int32_t n = 0; n < x_dims[0]; ++n) {
        for (int32_t h = 0; h < x_dims[1]; ++h) {
          for (int32_t w = 0; w < x_dims[2]; ++w) {
            for (int32_t c = 0; c < x_dims[3]; ++c) {
              const double error = abs_err + rel_err * base::abs(*x);
              EXPECT_NEAR(*x, *y, error) << "with index = [" << n << ", " << h
                                         << ", " << w << ", " << c << "]";
              x++;
              y++;
            }
          }
        }
      }
    } else {
      const int32_t size = base::GetShapeSize(x_dim_size, x_dims);
      for (int32_t i = 0; i < size; ++i) {
        const double error = abs_err + rel_err * base::abs(x[i]);
        EXPECT_NEAR(x[i], y[i], error);
      }
    }
  }
};

template<typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, false> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(
      const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
      const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size) {
    AssertSameDims(x_dims, x_dim_size, y_dims, y_dim_size);
    const int32_t size = base::GetShapeSize(x_dim_size, x_dims);
    for (int32_t i = 0; i < size; ++i) {
      ExpectEqual(x[i], y[i]);
    }
  }

  static void Near(
      const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
      const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size,
      const double rel_err, const double abs_err) {
    MACE_UNUSED(rel_err);
    MACE_UNUSED(abs_err);
    Equal(x, x_dims, x_dim_size, y, y_dims, y_dim_size);
  }
};

template<typename EXP_TYPE, typename RES_TYPE>
void ExpectTensorNear(
    const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
    const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size,
    const double rel_err = 1e-5, const double abs_err = 1e-8) {
  Expector<EXP_TYPE, RES_TYPE>::Near(x, x_dims, x_dim_size, y,
                                     y_dims, y_dim_size, rel_err, abs_err);
}

template<typename T>
void ExpectTensorNear(
    const T *x, const int32_t *x_dims, const uint32_t x_dim_size,
    const T *y, const int32_t *y_dims, const uint32_t y_dim_size,
    const double rel_err = 1e-5, const double abs_err = 1e-8) {
  Expector<T, T>::Near(x, x_dims, x_dim_size, y,
                       y_dims, y_dim_size, rel_err, abs_err);
}

template<typename EXP_TYPE, typename RES_TYPE>
void ExpectTensorSimilar(
    const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
    const RES_TYPE *y, const int32_t *y_dims, const uint32_t y_dim_size,
    const double rel_err = 1e-5) {
  AssertSameDims(x_dims, x_dim_size, y_dims, y_dim_size);
  const int32_t size = base::GetShapeSize(x_dim_size, x_dims);
  double dot_product = 0.0, x_norm = 0.0, y_norm = 0.0;
  for (int32_t i = 0; i < size; i++) {
    dot_product += x[i] * y[i];
    x_norm += x[i] * x[i];
    y_norm += y[i] * y[i];
  }
  double norm_product = base::sqrt(x_norm) * base::sqrt(y_norm);
  double error = rel_err * base::abs(dot_product);

  EXPECT_NEAR(dot_product, norm_product, error);
  PrintDims(x_dims, x_dim_size);
}

}  // namespace test
}  // namespace ops
}  // namespace micro

#endif  // MICRO_TEST_CCUTILS_MICRO_OPS_GTEST_UTILS_H_

