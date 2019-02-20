// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_UTILS_MATH_H_
#define MACE_UTILS_MATH_H_

#include <cmath>

#include <algorithm>
#include <vector>

#include "mace/utils/logging.h"

namespace mace {

template <typename Integer>
Integer RoundUp(Integer i, Integer factor) {
  return (i + factor - 1) / factor * factor;
}

template <typename Integer, uint32_t factor>
Integer RoundUpDiv(Integer i) {
  return (i + factor - 1) / factor;
}

// Partial specialization of function templates is not allowed
template <typename Integer>
Integer RoundUpDiv4(Integer i) {
  return (i + 3) >> 2;
}

template <typename Integer>
Integer RoundUpDiv8(Integer i) {
  return (i + 7) >> 3;
}

template <typename Integer>
Integer RoundUpDiv(Integer i, Integer factor) {
  return (i + factor - 1) / factor;
}

template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

template <typename Integer>
inline Integer Clamp(Integer in, Integer low, Integer high) {
  return std::max<Integer>(low, std::min<Integer>(in, high));
}

inline float ScalarSigmoid(float in) {
  if (in > 0) {
    return 1 / (1 + std::exp(-in));
  } else {
    float x = std::exp(in);
    return x / (x + 1.f);
  }
}

inline float ScalarTanh(float in) {
  if (in > 0) {
    float x = std::exp(-in);
    return -1.f + 2.f / (1.f + x * x);
  } else {
    float x = std::exp(in);
    return 1.f - 2.f / (1.f + x * x);
  }
}

template <typename SrcType, typename DstType>
std::vector<DstType> TransposeShape(const std::vector<SrcType> &shape,
                                    const std::vector<int> &dst_dims) {
  size_t shape_dims = shape.size();
  MACE_CHECK(shape_dims == dst_dims.size());
  std::vector<DstType> output_shape(shape_dims);
  for (size_t i = 0; i < shape_dims; ++i) {
    output_shape[i] = static_cast<DstType>(shape[dst_dims[i]]);
  }
  return output_shape;
}

}  // namespace mace

#endif  // MACE_UTILS_MATH_H_
