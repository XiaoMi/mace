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

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/split.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<RuntimeType D, typename T>
class SplitOp;

template<typename T>
class SplitOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit SplitOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 3)),
        checked_(false) {}

  void Validate() {
    if (axis_ < 0) {
      axis_ += this->Input(0)->dim_size();
    }
    auto has_df = Operation::GetOptionalArg<int>(
        "has_data_format", 0);
    if (has_df && this->Input(0)->dim_size() == 4) {
      if (axis_ == 3) axis_ = 1;
      else if (axis_ == 2) axis_ = 3;
      else if (axis_ == 1) axis_ = 2;
    }
    MACE_CHECK(this->OutputSize() >= 2)
      << "There must be at least two outputs for slicing";
    const Tensor *split_tensor =
        this->InputSize() == 2 ? this->Input(1) : nullptr;
    if ((this->Input(0)->dim(axis_) % this->OutputSize())) {
      MACE_CHECK(split_tensor != nullptr)
        << "To split input unequally, split sizes must be specified.";
    }
    if (split_tensor != nullptr) {
      index_t split_dim_size = split_tensor->dim_size();
      MACE_CHECK(split_dim_size == 1) << "Split input data must be 1D tensor, "
          << split_dim_size << "D data is got.";
      index_t num_splits = split_tensor->dim(0);
      MACE_CHECK(num_splits == this->OutputSize())
          << "Output size (" << this->OutputSize()
          << ") must be equal to split size ("
          << num_splits << ").";
      DataType split_dt = split_tensor->dtype();
      MACE_CHECK(split_dt == DataType::DT_INT32)
          << "Split tensor must have int32 datatype, "
          << "datatype " << split_dt << " is got.";
    } else {
      MACE_CHECK((this->Input(0)->dim(axis_) % this->OutputSize()) == 0)
        << "When split sizes are not specified, "
        << "outputs must split input equally.";
    }

    checked_ = true;
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    if (!checked_) Validate();
    const Tensor *input = this->Input(0);
    const Tensor *split_tensor =
        this->InputSize() == 2 ? this->Input(1) : nullptr;
    const std::vector<Tensor *> output_list = this->Outputs();
    const size_t outputs_count = output_list.size();
    const index_t input_channels = input->dim(axis_);
    std::vector<index_t> output_channels_list = std::vector<index_t>(
          outputs_count, input_channels / outputs_count);
    std::vector<std::vector<index_t>> output_shape_list;
    const std::vector<index_t> &input_shape = input->shape();
    if (split_tensor != nullptr) {
      const int32_t *split_data = split_tensor->data<int32_t>();
      output_shape_list = std::vector<std::vector<index_t>>(
          outputs_count, input_shape);
      for (size_t i = 0; i < outputs_count; ++i) {
        output_channels_list[i] = static_cast<index_t>(split_data[i]);
        output_shape_list[i][axis_] = output_channels_list[i];
      }
    } else {
      std::vector<index_t> output_shape(input->shape());
      output_shape[axis_] = output_channels_list[0];
      output_shape_list = std::vector<std::vector<index_t>>(
          outputs_count, output_shape);
    }
    std::vector<T *> output_ptrs(output_list.size(), nullptr);

    const index_t outer_size = std::accumulate(input_shape.begin(),
                                               input_shape.begin() + axis_,
                                               1,
                                               std::multiplies<index_t>());
    const index_t inner_size = std::accumulate(input_shape.begin() + axis_ + 1,
                                               input_shape.end(),
                                               1,
                                               std::multiplies<index_t>());
    for (size_t i = 0; i < outputs_count; ++i) {
      const std::vector<index_t> &output_shape = output_shape_list[i];
      MACE_RETURN_IF_ERROR(output_list[i]->Resize(output_shape));
      output_ptrs[i] = output_list[i]->mutable_data<T>();
    }
    const T *input_ptr = input->data<T>();

    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      index_t input_idx = outer_idx * input_channels * inner_size;
      index_t multiplier = outer_idx * inner_size;
      for (size_t i = 0; i < outputs_count; ++i) {
        index_t output_idx = multiplier * output_channels_list[i];
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptrs[i] + output_idx, input_ptr + input_idx,
                 output_channels_list[i] * inner_size * sizeof(T));
        } else {
          for (index_t k = 0; k < output_channels_list[i] * inner_size; ++k) {
            *(output_ptrs[i] + output_idx + k) = *(input_ptr + input_idx + k);
          }
        }
        input_idx += output_channels_list[i] * inner_size;
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int32_t axis_;
  bool checked_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class SplitOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit SplitOp(OpConstructContext *context)
      : Operation(context) {
    int32_t axis = Operation::GetOptionalArg<int>("axis", 3);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SplitKernel>(axis);
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

void RegisterSplit(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Split", SplitOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Split", SplitOp, RuntimeType::RT_CPU);

  MACE_REGISTER_GPU_OP(op_registry, "Split", SplitOp);

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Split")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<RuntimeType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
                }
                int axis = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                    *op, "axis", 3);
                if (axis != 3) return {RuntimeType::RT_CPU};

                for (int i = 0; i < op->output_size(); ++i) {
                  if (op->output_shape(i).dims_size() != 4 ||
                      op->output_shape(i).dims()[3] % 4 != 0) {
                    return {RuntimeType::RT_CPU};
                  }
                }

                return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
              }));
}

}  // namespace ops
}  // namespace mace
