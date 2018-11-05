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

#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ReshapeOp : public Operation {
 public:
  explicit ReshapeOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const std::vector<index_t> &input_shape = input->shape();
    int axis = Operation::GetOptionalArg<int>("reshape_axis", 0);
    int num_axes = Operation::GetOptionalArg<int>("num_axes", -1);
    MACE_CHECK(axis == 0 && num_axes == -1,
               "Only support axis = 0 and num_axes = -1");
    const Tensor *shape = this->Input(SHAPE);
    const index_t num_dims = shape->dim_size() == 0 ? 0 : shape->dim(0);
    Tensor::MappingGuard shape_guard(shape);
    const int32_t *shape_data = shape->data<int32_t>();

    int unknown_idx = -1;
    index_t product = 1;
    std::vector<index_t> out_shape;
    index_t n = 0;

    for (int i = 0; i < num_dims; ++i) {
      if (shape_data[i] == -1) {
        MACE_CHECK(unknown_idx == -1, "Only one input size may be -1");
        unknown_idx = i;
        out_shape.push_back(1);
      } else if (shape_data[i] == 0) {
        MACE_CHECK(shape_data[i] == 0, "Shape should be 0");
        out_shape.push_back(input_shape[i]);
        product *= input_shape[i];
      } else {
        MACE_CHECK(shape_data[i] > 0, "Shape must be non-negative: ",
                   shape_data[i]);
        if (shape_data[i] == 0) {
          MACE_CHECK(i < input->dim_size(),
                     "dims:0 out of input dims' range.");
          n = input->dim(i);
        } else {
          n = shape_data[i];
        }
        out_shape.push_back(n);
        product *= n;
      }
    }

    if (unknown_idx != -1) {
      MACE_CHECK(product != 0)
          << "Cannot infer shape if there is zero shape size.";
      const index_t missing = input->size() / product;
      MACE_CHECK(missing * product == input->size())
          << "Input size not match reshaped tensor size";
      out_shape[unknown_idx] = missing;
    }

    Tensor *output = this->Output(OUTPUT);
    output->ReuseTensorBuffer(*input);
    output->Reshape(out_shape);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterReshape(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::CPU, int32_t);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
