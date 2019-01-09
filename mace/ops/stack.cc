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
class StackOp : public Operation {
 public:
  explicit StackOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const std::vector<const Tensor *> &inputs = this->Inputs();
    Tensor *output = this->Output(0);
    MACE_CHECK(!inputs.empty(), "stack inputs are empty.");
    std::vector<index_t> input_shape = inputs[0]->shape();
    MACE_CHECK(axis_ >= -(inputs[0]->dim_size() + 1) &&
        axis_ < inputs[0]->dim_size() + 1,
               "axis out of bound.");
    if (axis_ < 0) {
      axis_ += inputs[0]->dim_size() + 1;
    }
    std::vector<index_t> output_shape = input_shape;
    output_shape.insert(output_shape.begin() + axis_, inputs.size());
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    // Some inputs_ may be in gpu memory, so add mapping here.
    std::vector<Tensor::MappingGuard> mappers;
    for (size_t i = 0; i < inputs.size(); ++i) {
      mappers.emplace_back(Tensor::MappingGuard(inputs[i]));
    }

    // Output is on host, no need to map data
    Tensor::MappingGuard output_guard(output);
    auto *output_data = output->mutable_data<T>();
    std::vector<const T *> input_data(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_data[i] = inputs[i]->data<T>();
    }

    index_t high_dim_elem_size =
        std::accumulate(input_shape.begin(), input_shape.begin() + axis_, 1,
                        std::multiplies<index_t>());
    index_t low_dim_elem_size =
        std::accumulate(input_shape.begin() + axis_, input_shape.end(), 1,
                        std::multiplies<index_t>());
    for (index_t h = 0; h < high_dim_elem_size; ++h) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        memcpy(output_data, input_data[i] + h * low_dim_elem_size,
               sizeof(T) * low_dim_elem_size);
        output_data += low_dim_elem_size;
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int axis_;
};

void RegisterStack(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Stack", StackOp, DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Stack", StackOp, DeviceType::CPU, int32_t);
}

}  // namespace ops
}  // namespace mace
