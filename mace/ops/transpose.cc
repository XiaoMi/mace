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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/common/transpose.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class TransposeOp : public Operation {
 public:
  explicit TransposeOp(OpConstructContext *context)
      : Operation(context),
        dims_(Operation::GetRepeatedArgs<int>("dims")) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();
    MACE_CHECK((input_shape.size() == 4 && dims_.size() == 4) ||
                   (input_shape.size() == 3 && dims_.size() == 3) ||
                   (input_shape.size() == 2 && dims_.size() == 2),
               "rank should be 2, 3 or 4");
    std::vector<index_t> output_shape;
    for (size_t i = 0; i < dims_.size(); ++i) {
      output_shape.push_back(input_shape[dims_[i]]);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    return Transpose(&context->device()->cpu_runtime()->thread_pool(),
                     input_data, input->shape(), dims_, output_data);
  }

 private:
  std::vector<int> dims_;
};

void RegisterTranspose(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Transpose", TransposeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Transpose", TransposeOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
