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


#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

namespace mace {
namespace ops {

template <RuntimeType D, class T>
class FillOp;

template <class T>
class FillOp<RuntimeType::RT_CPU, T> : public Operation {
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
    const int32_t *shape_data = shape->data<int32_t>();

    std::vector<index_t> output_shape;
    for (index_t i = 0; i < num_dims; ++i) {
      MACE_CHECK(shape_data[i] > 0, "Shape must be non-negative: ",
                 shape_data[i]);
      output_shape.push_back(shape_data[i]);
    }

    const T *value_data = value->data<T>();
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    T *output_data = output->mutable_data<T>();

    std::fill(output_data, output_data + output->size(), *value_data);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(SHAPE, VALUE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterFill(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Fill", FillOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Fill", FillOp, RuntimeType::RT_CPU);
}

}  // namespace ops
}  // namespace mace
