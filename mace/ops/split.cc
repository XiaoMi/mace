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

#include <functional>
#include <memory>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/split.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class SplitOp;

template<typename T>
class SplitOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SplitOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 3)),
        checked_(false) {}

  void Validate() {
    auto has_df = Operation::GetOptionalArg<int>(
        "has_data_format", 0);
    if (has_df && this->Input(0)->dim_size() == 4) {
      if (axis_ == 3) axis_ = 1;
      else if (axis_ == 2) axis_ = 3;
      else if (axis_ == 1) axis_ = 2;
    }
    MACE_CHECK(this->OutputSize() >= 2)
      << "There must be at least two outputs for slicing";
    MACE_CHECK((this->Input(0)->dim(axis_) % this->OutputSize()) == 0)
      << "Outputs do not split input equally.";
    checked_ = true;
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    if (!checked_) Validate();
    const Tensor *input = this->Input(0);
    const std::vector<Tensor *> output_list = this->Outputs();
    const index_t input_channels = input->dim(axis_);
    const size_t outputs_count = output_list.size();
    const index_t output_channels = input_channels / outputs_count;
    std::vector<T *> output_ptrs(output_list.size(), nullptr);
    std::vector<index_t> output_shape(input->shape());
    output_shape[axis_] = output_channels;

    const index_t outer_size = std::accumulate(output_shape.begin(),
                                               output_shape.begin() + axis_,
                                               1,
                                               std::multiplies<index_t>());
    const index_t inner_size = std::accumulate(output_shape.begin() + axis_ + 1,
                                               output_shape.end(),
                                               1,
                                               std::multiplies<index_t>());
    for (size_t i = 0; i < outputs_count; ++i) {
      MACE_RETURN_IF_ERROR(output_list[i]->Resize(output_shape));
      output_ptrs[i] = output_list[i]->mutable_data<T>();
    }
    const T *input_ptr = input->data<T>();

    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      index_t input_idx = outer_idx * input_channels * inner_size;
      index_t output_idx = outer_idx * output_channels * inner_size;
      for (size_t i = 0; i < outputs_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptrs[i] + output_idx, input_ptr + input_idx,
                 output_channels * inner_size * sizeof(T));
        } else {
          for (index_t k = 0; k < output_channels * inner_size; ++k) {
            *(output_ptrs[i] + output_idx + k) = *(input_ptr + input_idx + k);
          }
        }
        input_idx += output_channels * inner_size;
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int32_t axis_;
  bool checked_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class SplitOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit SplitOp(OpConstructContext *context)
      : Operation(context) {
    int32_t axis = Operation::GetOptionalArg<int>("axis", 3);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SplitKernel<T>>(axis);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    MACE_CHECK(this->OutputSize() >= 2)
      << "There must be at least two outputs for slicing";
    const Tensor *input = this->Input(0);
    const std::vector<Tensor *> output_list = this->Outputs();
    int32_t axis = Operation::GetOptionalArg<int>("axis", 3);
    MACE_CHECK((input->dim(axis) % this->OutputSize()) == 0)
      << "Outputs do not split input equally.";
    return kernel_->Compute(context, input, output_list);
  }

 private:
  std::unique_ptr<OpenCLSplitKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterSplit(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Split", SplitOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Split", SplitOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Split", SplitOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Split")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return {DeviceType::CPU, DeviceType::GPU};
                }
                int axis = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                    *op, "axis", 3);
                if (axis != 3 || op->output_shape(0).dims_size() != 4 ||
                    (op->output_shape(0).dims()[3] % 4 != 0)) {
                  return {DeviceType::CPU};
                }
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
