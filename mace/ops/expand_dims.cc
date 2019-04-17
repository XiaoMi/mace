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


#include "mace/core/operator.h"
#include "mace/ops/common/transpose.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ExpandDimsOp;

template<typename T>
class ExpandDimsOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ExpandDimsOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    index_t input_dims_size = input->dim_size();
    if (axis_ < 0) {
      axis_ += input_dims_size + 1;
    }
    MACE_CHECK(axis_ >= 0 && axis_ <= input_dims_size,
               "axis is out of bound: ", axis_);
    const std::vector<index_t> input_shape = input->shape();
    std::vector<index_t> output_shape(input_shape);
    output_shape.insert(output_shape.begin() + axis_, 1);

    bool has_data_format = Operation::GetOptionalArg<int>(
        "has_data_format", 0) == 1;
    if (has_data_format && output_shape.size() == 4) {
      // only tensorflow support expand dim, so the default format is NHWC
      // transform NHWC to NCHW
      auto t_output_shape = TransposeShape<int64_t, int64_t>(output_shape,
                                                             {0, 3, 1, 2});
      output->Resize(t_output_shape);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      auto input_data = input->data<T>();
      auto output_data = output->mutable_data<T>();

      Transpose(&context->device()->cpu_runtime()->thread_pool(),
                input_data, output_shape, {0, 3, 1, 2}, output_data);
    } else {
      output->Resize(output_shape);
      Tensor::MappingGuard input_guard(input);
      auto input_data = input->data<T>();
      output->Copy<T>(input_data, input->size());
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int axis_;
};

void RegisterExpandDims(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ExpandDims", ExpandDimsOp,
                   DeviceType::CPU, float);

  MACE_REGISTER_OP(op_registry, "ExpandDims", ExpandDimsOp,
                   DeviceType::CPU, int32_t);
}

}  // namespace ops
}  // namespace mace
