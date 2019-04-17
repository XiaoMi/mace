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

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SliceOp;

template <typename T>
class SliceOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SliceOp(OpConstructContext *context)
      : Operation(context),
        axes_(Operation::GetRepeatedArgs<int>("axes")),
        starts_(Operation::GetRepeatedArgs<int>("starts")),
        ends_(Operation::GetRepeatedArgs<int>("ends")) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    const index_t rank = input->dim_size();
    MACE_CHECK(rank >= 1)
      << "The input dim size should >= 1";
    const index_t input_dim = input->dim(rank - 1);
    MACE_CHECK(starts_.size() == 1 && ends_.size() == 1 && axes_.size() == 1,
               "only support slicing at one axis.");
    MACE_CHECK(axes_[0] == -1 || axes_[0] == rank - 1,
               "only support slicing at the last axis.");
    MACE_CHECK(starts_[0] < input_dim && starts_[0] >= 0
                   && ends_[0] >= 0
                   && ends_[0] <= input_dim)
      << "The starts and ends caused over range error.";
    const index_t offset = starts_[0];
    const index_t output_dim = ends_[0] - starts_[0];
    MACE_CHECK(output_dim >= 0, "output_dim should >= 0");

    const index_t  frames =
        std::accumulate(input->shape().begin(), input->shape().end() - 1, 1,
                        std::multiplies<index_t>());

    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 1] = output_dim;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    for (index_t i = 0; i < frames; ++i) {
      const T *input_base =
          input_data + i * input_dim + offset;
      T *output_base =
          output_data + i * output_dim;
      memcpy(output_base, input_base, output_dim * sizeof(T));
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> axes_;
  std::vector<int> starts_;
  std::vector<int> ends_;
};

void RegisterSlice(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Slice", SliceOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
