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

#include <algorithm>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class GatherOp;

template <>
class GatherOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit GatherOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)),
        y_(Operation::GetOptionalArg<float>("y", 1.0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *params = this->Input(PARAMS);
    const Tensor *indices = this->Input(INDICES);
    Tensor *output = this->Output(OUTPUT);
    std::vector<index_t> output_shape;
    if (axis_ < 0) {
      axis_ += params->dim_size();
    }
    MACE_CHECK(axis_ >= 0 && axis_ < params->dim_size(),
               "axis is out of bound: ", axis_);
    output_shape.insert(output_shape.end(), params->shape().begin(),
                        params->shape().begin() + axis_);
    output_shape.insert(output_shape.end(), indices->shape().begin(),
                        indices->shape().end());
    output_shape.insert(output_shape.end(),
                        params->shape().begin() + (axis_ + 1),
                        params->shape().end());
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard indices_guard(indices);
    Tensor::MappingGuard params_guard(params);
    Tensor::MappingGuard output_guard(output);
    const int32_t *indices_data = indices->data<int32_t>();
    const float *params_data = params->data<float>();
    float *output_data = output->mutable_data<float>();

    index_t axis_dim_size = params->dim(axis_);
    index_t lhs_size = std::accumulate(params->shape().begin(),
                                       params->shape().begin() + axis_, 1,
                                       std::multiplies<index_t>());
    index_t rhs_size =
        std::accumulate(params->shape().begin() + (axis_ + 1),
                        params->shape().end(), 1, std::multiplies<index_t>());
    index_t index_size = indices->size();

#pragma omp parallel for collapse(2)
    for (index_t l = 0; l < lhs_size; ++l) {
      for (index_t idx = 0; idx < index_size; ++idx) {
        MACE_ASSERT(indices_data[idx] < axis_dim_size, "idx out of bound: ",
                    indices_data[idx]);
        memcpy(
            output_data + ((l * index_size) + idx) * rhs_size,
            params_data + ((l * axis_dim_size) + indices_data[idx]) * rhs_size,
            sizeof(float) * rhs_size);
      }
    }

    if (std::fabs(y_ - 1.0) > 1e-6) {
#pragma omp parallel for
      for (index_t i = 0; i < output->size(); ++i) {
        output_data[i] *= y_;
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int axis_;
  float y_;
  MACE_OP_INPUT_TAGS(PARAMS, INDICES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterGather(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Gather", GatherOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
