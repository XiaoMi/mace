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

#include <vector>

#include "mace/core/operator.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ReshapeOp : public Operation {
 public:
  explicit ReshapeOp(OpConstructContext *context)
      : Operation(context),
        has_df_(Operation::GetOptionalArg<int>("has_data_format", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *shape = this->Input(SHAPE);
    const index_t num_dims = shape->dim_size() == 0 ? 0 : shape->dim(0);
    Tensor::MappingGuard shape_guard(shape);
    const int32_t *shape_data = shape->data<int32_t>();

    int unknown_idx = -1;
    index_t product = 1;
    std::vector<index_t> out_shape(num_dims);
    index_t n = 0;

    for (int i = 0; i < num_dims; ++i) {
      if (shape_data[i] == -1) {
        MACE_CHECK(unknown_idx == -1, "Only one input size may be -1");
        unknown_idx = i;
        out_shape[i] = 1;
      } else {
        MACE_CHECK(shape_data[i] >= 0, "Shape must be non-negative: ",
                   shape_data[i]);
        if (shape_data[i] == 0) {
          MACE_CHECK(i < input->dim_size(),
                     "dims:0 out of input dims' range.");
          n = input->dim(i);
        } else {
          n = shape_data[i];
        }
        out_shape[i] = n;
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
    // NHWC -> NCHW

    if (has_df_ && D == DeviceType::CPU
        && out_shape.size() == 4 && shape->is_weight()) {
      std::vector<int> dst_dims = {0, 3, 1, 2};
      std::vector<index_t> trans_shape = TransposeShape<index_t, index_t>(
          out_shape, dst_dims);
      out_shape = trans_shape;
    }

    output->ReuseTensorBuffer(*input);
    output->Reshape(out_shape);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool has_df_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterReshape(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   DeviceType::CPU, int32_t);
}

}  // namespace ops
}  // namespace mace
