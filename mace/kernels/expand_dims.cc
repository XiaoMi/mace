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


#include "mace/core/operator.h"

namespace mace {
namespace kernels {

template <DeviceType D, class T>
class ExpandDimsOp;

template <typename T>
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
    if ( axis_ < 0 ) {
      axis_ += input_dims_size + 1;
    }
    MACE_CHECK(axis_ >= 0 && axis_ <= input_dims_size,
               "axis is out of bound: ", axis_);
    const std::vector<index_t> input_shape = input->shape();
    std::vector<index_t> output_shape;
    output_shape.insert(output_shape.end(), input_shape.begin(),
                        input_shape.begin() + axis_);
    output_shape.insert(output_shape.end(), 1);
    output_shape.insert(output_shape.end(), input_shape.begin() + axis_,
                        input_shape.end());

    output->ReuseTensorBuffer(*input);
    output->Reshape(output_shape);

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

  MACE_REGISTER_OP(op_registry, "ExpandDims", ExpandDimsOp,
                   DeviceType::CPU, uint8_t);
}

}  // namespace kernels
}  // namespace mace
