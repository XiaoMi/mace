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

namespace mace {
namespace ops {

template <DeviceType D, class T>
class FillOp;

template <>
class FillOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit FillOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *shape = this->Input(SHAPE);
    const Tensor *value = this->Input(VALUE);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(shape->dim_size() == 1, "Shape must be 1-D");
    const index_t num_dims = shape->dim(0);
    Tensor::MappingGuard shape_guard(shape);
    const int32_t *shape_data = shape->data<int32_t>();

    std::vector<index_t> output_shape;
    for (index_t i = 0; i < num_dims; ++i) {
      MACE_CHECK(shape_data[i] > 0, "Shape must be non-negative: ",
                 shape_data[i]);
      output_shape.push_back(shape_data[i]);
    }

    Tensor::MappingGuard value_guard(value);
    const float *value_data = value->data<float>();

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard output_guard(output);
    float *output_data = output->mutable_data<float>();

    std::fill(output_data, output_data + output->size(), *value_data);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(SHAPE, VALUE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterFill(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Fill", FillOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
