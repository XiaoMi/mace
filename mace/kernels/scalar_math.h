// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_SCALAR_MATH_H_
#define MACE_KERNELS_SCALAR_MATH_H_

#include <algorithm>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/kernels/eltwise.h"

namespace mace {
namespace kernels {

template <typename T, typename DstType>
void ScalarEltwise(const T* in0,
                   const T* in1,
                   const EltwiseType type,
                   const std::vector<float> &coeff,
                   const bool swapped,
                   DstType* out) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
        out[0] = in0[0] + in1[0];
      } else {
        MACE_CHECK(coeff.size() == 2,
                   "sum's coeff params' size should be 2.");
        if (swapped)
          out[0] = in0[0] * coeff[1] + in1[0] * coeff[0];
        else
          out[0] = in0[0] * coeff[0] + in1[0] * coeff[1];
      }
      break;
    case SUB:
      if (swapped)
        out[0] = in1[0] - in0[0];
      else
        out[0] = in0[0] - in1[0];
      break;
    case PROD:
      out[0] = in0[0] * in1[0];
      break;
    case DIV:
      if (swapped)
        out[0] = in1[0] / in0[0];
      else
        out[0] = in0[0] / in1[0];
      break;
    case MIN:
      out[0] = std::min(in1[0], in0[0]);
      break;
    case MAX:
      out[0] = std::max(in1[0], in0[0]);
      break;
    case SQR_DIFF:
      out[0] = std::pow(in1[0] - in0[0], 2.f);
      break;
    case POW:
      out[0] = std::pow(in0[0], in1[0]);
      break;
    case EQUAL:
      out[0] = in1[0] == in0[0];
      break;
    case NEG:
      out[0] = -in0[0];
      break;
    case ABS:
      out[0] = in0[0] > 0 ? in0[0] : -in0[0];
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}


template <DeviceType D, typename T>
struct ScalarMathFunctor {
  explicit ScalarMathFunctor(const EltwiseType type,
                             const std::vector<float> &coeff,
                             const float scalar_input,
                             const int32_t scalar_input_index)
      : type_(type),
        coeff_(coeff),
        scalar_input_(scalar_input),
        scalar_input_index_(scalar_input_index) {}

  MaceStatus operator()(const std::vector<const Tensor *> &inputs,
                        Tensor *output,
                        StatsFuture *future) {
    const Tensor* input0 = inputs[0];
    const Tensor* input1 = (inputs.size() >= 2) ? inputs[1] : nullptr;
    MACE_CHECK(input0->dim_size() <= 1 && input0->size() == 1,
               "not support input dim size") << input0->dim_size();
    Tensor::MappingGuard in0_guard(input0);
    const T* in0 = input0->data<T>();
    auto v = static_cast<T>(scalar_input_);
    const T* in1 = &v;
    Tensor::MappingGuard in1_guard(input1);
    if (input1) {
      MACE_CHECK(input1->dim_size() == 0);
      in1 = input1->data<T>();
    }
    if (input0->dim_size() > 0) {
      MACE_RETURN_IF_ERROR(output->Resize(input0->shape()));
    } else {
      output->Resize({});
    }

    Tensor::MappingGuard output_guard(output);
    bool swapped = scalar_input_index_ == 0;

    if (IsLogicalType(type_)) {
      int32_t* out = output->mutable_data<int32_t>();
      ScalarEltwise<T, int32_t>(in0,
                                in1,
                                type_,
                                coeff_,
                                swapped,
                                out);
    } else {
      T* out = output->mutable_data<T>();
      ScalarEltwise<T, T>(in0,
                          in1,
                          type_,
                          coeff_,
                          swapped,
                          out);
    }

    SetFutureDefaultWaitFn(future);
    return MACE_SUCCESS;
  }

  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SCALAR_MATH_H_
