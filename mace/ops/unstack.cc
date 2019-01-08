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

#include <algorithm>
#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class UnstackOp : public Operation {
 public:
  explicit UnstackOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    const std::vector<Tensor *> outputs = this->Outputs();
    std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(axis_ >= -(input->dim_size()) && axis_ < input->dim_size(),
               "axis out of bound.");
    if (axis_ < 0) {
      axis_ += input->dim_size();
    }
    MACE_CHECK(static_cast<index_t>(outputs.size()) == input_shape[axis_],
               "output size not equal input_shape[axis]");

    std::vector<index_t> output_shape = input_shape;
    output_shape.erase(output_shape.begin() + axis_);

    std::vector<T *> output_data(outputs.size(), nullptr);
    for (index_t i = 0; i < input_shape[axis_]; ++i) {
      MACE_RETURN_IF_ERROR(outputs[i]->Resize(output_shape));
      output_data[i] = outputs[i]->mutable_data<T>();
    }
    const T *input_data = input->data<T>();

    index_t high_dim_elem_size =
        std::accumulate(input_shape.begin(), input_shape.begin() + axis_, 1,
                        std::multiplies<index_t>());
    index_t low_dim_elem_size =
        std::accumulate(input_shape.begin() + axis_ + 1, input_shape.end(), 1,
                        std::multiplies<index_t>());

    for (index_t h = 0; h < high_dim_elem_size; ++h) {
      int input_idx = h * input_shape[axis_] * low_dim_elem_size;
      int output_idx = h * low_dim_elem_size;
      for (index_t i = 0; i < input_shape[axis_]; ++i) {
        memcpy(output_data[i] + output_idx, input_data + input_idx,
               sizeof(T) * low_dim_elem_size);
        input_idx += low_dim_elem_size;
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int axis_;
};

void RegisterUnstack(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Unstack", UnstackOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Unstack", UnstackOp,
                   DeviceType::CPU, int32_t);
}

}  // namespace ops
}  // namespace mace
