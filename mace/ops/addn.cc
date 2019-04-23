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

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/addn.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class AddNOp;

template <>
class AddNOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit AddNOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(inputs_[0]));
    const index_t size = output->size();

    Tensor::MappingGuard output_guard(output);
    auto output_data = output->mutable_data<float>();
    memset(output_data, 0, size * sizeof(float));

    for (auto &input : inputs_) {
      Tensor::MappingGuard input_guard(input);
      auto input_data = input->data<float>();

      for (index_t j = 0; j < size; ++j) {
        output_data[j] += input_data[j];
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class AddNOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit AddNOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::AddNKernel<T>>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    Tensor *output_tensor = this->Output(0);
    size_t n = this->inputs_.size();
    for (size_t i = 1; i < n; ++i) {
      MACE_CHECK(inputs_[0]->dim_size() == inputs_[i]->dim_size());
      MACE_CHECK(inputs_[0]->size() == inputs_[i]->size())
        << "Input 0: " << MakeString(inputs_[0]->shape())
        << ", size: " << inputs_[0]->size() << ". Input " << i << ": "
        << MakeString(inputs_[i]->shape()) << ", size: " << inputs_[i]->size();
    }

    return kernel_->Compute(context, inputs_, output_tensor);
  }

 private:
  std::unique_ptr<OpenCLAddNKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterAddN(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("AddN")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                int has_data_format =
                    ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                        *op, "has_data_format", 0);
                if (!has_data_format ||
                    op->output_shape(0).dims_size() != 4) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace
