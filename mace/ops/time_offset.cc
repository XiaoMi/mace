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

// This Op is for offset descriptor in Kaldi.
// It defines time offset.

#include <functional>
#include <memory>

#include "mace/core/operator.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class TimeOffsetOp;

template <typename T>
class TimeOffsetOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit TimeOffsetOp(OpConstructContext *context)
      : Operation(context),
        offset_(Operation::GetOptionalArg<int>("offset", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    index_t rank = input->dim_size();
    MACE_CHECK(rank >= 2, "input's rank should >= 2.");
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t frames = input_shape[rank - 2];
    const index_t input_dim = input_shape[rank - 1];
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t i = 0; i < batch; ++i) {
      for (index_t j = 0; j < frames; ++j) {
        index_t time_index = offset_ + j;
        index_t index = Clamp<index_t>(time_index, 0, frames - 1);
        T *output_base = output_data + (i * frames + j) * input_dim;
        const T *input_base = input_data + (i * frames + index) * input_dim;
        memcpy(output_base, input_base, input_dim * sizeof(T));
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int offset_;
};

void RegisterTimeOffset(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "TimeOffset", TimeOffsetOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
