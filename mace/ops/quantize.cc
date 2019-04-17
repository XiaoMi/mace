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

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/core/quantize.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class QuantizeOp;

template<>
class QuantizeOp<DeviceType::CPU, uint8_t> : public Operation {
 public:
  explicit QuantizeOp(OpConstructContext *context)
      : Operation(context),
        non_zero_(
            static_cast<bool>(Operation::GetOptionalArg<int>("non_zero", 0))),
        find_range_every_time_(static_cast<bool>(Operation::GetOptionalArg<int>(
            "find_range_every_time",
            0))),
        quantize_util_(&context->device()->cpu_runtime()->thread_pool()) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    uint8_t *output_data = output->mutable_data<uint8_t>();
    if (!find_range_every_time_ && output->scale() > 0.f) {
      quantize_util_.QuantizeWithScaleAndZeropoint(input_data,
                                                   input->size(),
                                                   output->scale(),
                                                   output->zero_point(),
                                                   output_data);
    } else {
      float scale;
      int32_t zero_point;
      quantize_util_.Quantize(input_data,
                              input->size(),
                              non_zero_,
                              output_data,
                              &scale,
                              &zero_point);
      output->SetScale(scale);
      output->SetZeroPoint(zero_point);
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool non_zero_;
  bool find_range_every_time_;
  QuantizeUtil<uint8_t> quantize_util_;
};

template<DeviceType D, class T>
class DequantizeOp;

template<typename T>
class DequantizeOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit DequantizeOp(OpConstructContext *context)
      : Operation(context),
        quantize_util_(&context->device()->cpu_runtime()->thread_pool()) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    float *output_data = output->mutable_data<float>();
    quantize_util_.Dequantize(input_data,
                              input->size(),
                              input->scale(),
                              input->zero_point(),
                              output_data);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  QuantizeUtil<T> quantize_util_;
};

void RegisterQuantize(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Quantize", QuantizeOp,
                   DeviceType::CPU, uint8_t);
}

void RegisterDequantize(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Dequantize", DequantizeOp,
                   DeviceType::CPU, uint8_t);
  MACE_REGISTER_OP(op_registry, "Dequantize", DequantizeOp,
                   DeviceType::CPU, int32_t);
}
}  // namespace ops
}  // namespace mace
