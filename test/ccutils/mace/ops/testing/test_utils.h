// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#ifndef MACE_OPS_TESTING_TEST_UTILS_H_
#define MACE_OPS_TESTING_TEST_UTILS_H_

#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <functional>
#include <vector>

#include "mace/core/tensor.h"
#include "gtest/gtest.h"

namespace mace {
namespace ops {
namespace test {

template<typename T>
void GenerateRandomRealTypeData(const std::vector<index_t> &shape,
                                T *res,
                                bool positive = true) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());

  if (DataTypeToEnum<T>::value == DT_HALF) {
    std::generate(res, res + size, [&gen, &nd, positive] {
      return half_float::half_cast<half>(positive ? std::abs(nd(gen))
                                                  : nd(gen));
    });
  } else {
    std::generate(res, res + size, [&gen, &nd, positive] {
      return positive ? std::abs(nd(gen)) : nd(gen);
    });
  }
}

template<typename T>
void GenerateRandomRealTypeData(const std::vector<index_t> &shape,
                                std::vector<T> *res,
                                bool positive = true) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());
  res->resize(size);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    std::generate(res->begin(), res->end(), [&gen, &nd, positive] {
      return half_float::half_cast<half>(positive ? std::abs(nd(gen))
                                                  : nd(gen));
    });
  } else {
    std::generate(res->begin(), res->end(), [&gen, &nd, positive] {
      return positive ? std::abs(nd(gen)) : nd(gen);
    });
  }
}

template<typename T>
void GenerateRandomIntTypeData(const std::vector<index_t> &shape,
                               T *res,
                               const T a = 0,
                               const T b = std::numeric_limits<T>::max()) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> nd(a, b);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());

  std::generate(res, res + size, [&gen, &nd] { return nd(gen); });
}

template<typename T>
void GenerateRandomIntTypeData(const std::vector<index_t> &shape,
                               std::vector<T> *res,
                               const T a = 0,
                               const T b = std::numeric_limits<T>::max()) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> nd(a, b);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());
  res->resize(size);

  std::generate(res->begin(), res->end(), [&gen, &nd] { return nd(gen); });
}

template<typename T>
std::vector<T> VectorStaticCast(const std::vector<float> &&src) {
  std::vector<T> dest;
  dest.reserve(src.size());
  for (float f : src) {
    dest.push_back(static_cast<T>(f));
  }
  return std::move(dest);
}

inline bool IsSameSize(const Tensor &x, const Tensor &y) {
  if (x.dim_size() != y.dim_size()) return false;
  for (int d = 0; d < x.dim_size(); ++d) {
    if (x.dim(d) != y.dim(d)) return false;
  }
  return true;
}

inline std::string ShapeToString(const Tensor &x) {
  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < x.dim_size(); i++) {
    if (i > 0) stream << ",";
    int64_t dim = x.dim(i);
    if (dim < 0) {
      stream << "?";
    } else {
      stream << dim;
    }
  }
  stream << "]";
  return std::string(stream.str());
}

template<typename T>
struct is_floating_point_type {
  static const bool value = std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, half>::value;
};

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

inline void AssertSameDims(const Tensor &x, const Tensor &y) {
  ASSERT_TRUE(IsSameSize(x, y)) << "x.shape " << ShapeToString(x) << " vs "
                                << "y.shape " << ShapeToString(y);
}

template<typename EXP_TYPE,
  typename RES_TYPE,
  bool is_fp = is_floating_point_type<EXP_TYPE>::value>
struct Expector;

// Partial specialization for float and double.
template<typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, true> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(const Tensor &x, const Tensor &y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    for (int i = 0; i < x.size(); ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }

  static void Near(const Tensor &x,
                   const Tensor &y,
                   const double rel_err,
                   const double abs_err) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    if (x.dim_size() == 4) {
      for (int n = 0; n < x.dim(0); ++n) {
        for (int h = 0; h < x.dim(1); ++h) {
          for (int w = 0; w < x.dim(2); ++w) {
            for (int c = 0; c < x.dim(3); ++c) {
              const double error = abs_err + rel_err * std::abs(*a);
              EXPECT_NEAR(*a, *b, error) << "with index = [" << n << ", " << h
                                         << ", " << w << ", " << c << "]";
              a++;
              b++;
            }
          }
        }
      }
    } else {
      for (int i = 0; i < x.size(); ++i) {
        const double error = abs_err + rel_err * std::abs(a[i]);
        EXPECT_NEAR(a[i], b[i], error) << "a = " << a << " b = " << b
                                       << " index = " << i;
      }
    }
  }
};

template<typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, false> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(const Tensor &x, const Tensor &y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    for (int i = 0; i < x.size(); ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }

  static void Near(const Tensor &x,
                   const Tensor &y,
                   const double rel_err,
                   const double abs_err) {
    MACE_UNUSED(rel_err);
    MACE_UNUSED(abs_err);
    Equal(x, y);
  }
};

template<typename T>
void ExpectTensorNear(const Tensor &x,
                      const Tensor &y,
                      const double rel_err = 1e-5,
                      const double abs_err = 1e-8) {
  Expector<T, T>::Near(x, y, rel_err, abs_err);
}

template<typename EXP_TYPE, typename RES_TYPE>
void ExpectTensorNear(const Tensor &x,
                      const Tensor &y,
                      const double rel_err = 1e-5,
                      const double abs_err = 1e-8) {
  Expector<EXP_TYPE, RES_TYPE>::Near(x, y, rel_err, abs_err);
}

template<typename T>
void ExpectTensorSimilar(const Tensor &x,
                         const Tensor &y,
                         const double rel_err = 1e-5) {
  AssertSameDims(x, y);
  Tensor::MappingGuard x_mapper(&x);
  Tensor::MappingGuard y_mapper(&y);
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  double dot_product = 0.0, x_norm = 0.0, y_norm = 0.0;
  for (index_t i = 0; i < x.size(); i++) {
    dot_product += x_data[i] * y_data[i];
    x_norm += x_data[i] * x_data[i];
    y_norm += y_data[i] * y_data[i];
  }
  double norm_product = sqrt(x_norm) * sqrt(y_norm);
  double error = rel_err * std::abs(dot_product);

  EXPECT_NEAR(dot_product, norm_product, error)
            << "Shape " << ShapeToString(x);
}

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_TESTING_TEST_UTILS_H_

