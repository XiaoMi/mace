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

#ifndef MACE_KERNELS_QUANTIZE_H_
#define MACE_KERNELS_QUANTIZE_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"
#include "mace/utils/quantize.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct QuantizeFunctor;

template<>
struct QuantizeFunctor<CPU, uint8_t> : OpKernel {
  explicit QuantizeFunctor(OpKernelContext *context) : OpKernel(context) {}

  MaceStatus operator()(const Tensor *input,
                        const bool non_zero,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    uint8_t *output_data = output->mutable_data<uint8_t>();
    if (output->scale() > 0.f) {
      QuantizeWithScaleAndZeropoint(input_data,
                                    input->size(),
                                    output->scale(),
                                    output->zero_point(),
                                    output_data);
    } else {
      float scale;
      int32_t zero_point;
      Quantize(input_data,
               input->size(),
               non_zero,
               output_data,
               &scale,
               &zero_point);
      output->SetScale(scale);
      output->SetZeroPoint(zero_point);
    }

    return MACE_SUCCESS;
  }
};

template<DeviceType D, typename T>
struct DequantizeFunctor;

template<>
struct DequantizeFunctor<CPU, uint8_t> : OpKernel {
  explicit DequantizeFunctor(OpKernelContext *context) : OpKernel(context) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const uint8_t *input_data = input->data<uint8_t>();
    float *output_data = output->mutable_data<float>();
    Dequantize(input_data,
               input->size(),
               input->scale(),
               input->zero_point(),
               output_data);

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_QUANTIZE_H_
