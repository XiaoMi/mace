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

#include <unordered_set>
#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SqueezeOp : public Operation {
 public:
  explicit SqueezeOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetRepeatedArgs<int>("axis", {})) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    std::vector<index_t> output_shape;
    std::unordered_set<int> axis_set(axis_.begin(), axis_.end());
    for (int i = 0; i < input->dim_size(); ++i) {
      if (input->dim(i) > 1
          || (!axis_set.empty() && axis_set.find(i) == axis_set.end())) {
        output_shape.push_back(input->dim(i));
      }
    }
    output->ReuseTensorBuffer(*input);
    output->Reshape(output_shape);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> axis_;
};

void RegisterSqueeze(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::CPU, uint8_t);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
