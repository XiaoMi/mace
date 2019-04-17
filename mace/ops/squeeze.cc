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
        axis_(Operation::GetRepeatedArgs<int>("axis", {})),
        checked_(false) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    if (!checked_ && D == DeviceType::CPU
        && DataTypeToEnum<T>::value != DT_UINT8) {
      auto has_df = Operation::GetOptionalArg<int>(
          "has_data_format", 0);
      if (has_df && this->Input(0)->dim_size() == 4) {
        if (axis_.size() == 2 && axis_[0] == 1 && axis_[1] == 2) {
          axis_[0] = 2;
          axis_[1] = 3;
        }
      }
      checked_ = true;
    }
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
  bool checked_;
};

void RegisterSqueeze(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::CPU, float);
#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "Squeeze", SqueezeOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Squeeze")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                if (op->output_shape(0).dims_size() != 2 &&
                    op->output_shape(0).dims_size() != 4) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace
