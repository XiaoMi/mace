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

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class PadContextOp;

template <typename T>
class PadContextOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit PadContextOp(OpConstructContext *context)
      : Operation(context),
        left_context_(Operation::GetOptionalArg<int>("left_context", 0)),
        right_context_(Operation::GetOptionalArg<int>("right_context", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    index_t rank = input->dim_size();
    MACE_CHECK(rank >= 2, "input's rank should >= 2.");
    MACE_CHECK(left_context_ >= 0 && right_context_ >= 0,
               "left context and right context should be greater than zero");
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t chunk = input_shape[rank - 2];
    const index_t dim = input_shape[rank - 1];
    const index_t output_chunk = chunk + left_context_ + right_context_;
    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 2] = output_chunk;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    for (index_t i = 0; i < batch; ++i) {
      T *out_base = output_data + i * output_chunk * dim;
      const T *in_base = input_data + i * chunk * dim;
      for (index_t j = 0; j < left_context_; ++j) {
        memcpy(out_base + j * dim, in_base, dim * sizeof(T));
      }
      out_base = out_base + left_context_ * dim;
      memcpy(out_base, in_base, chunk * dim * sizeof(T));
      out_base = out_base + chunk * dim;
      in_base = in_base + (chunk -1) * dim;
      for (index_t j = 0; j < right_context_; ++j) {
        memcpy(out_base + j * dim, in_base, dim * sizeof(T));
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int left_context_;
  int right_context_;
};

void RegisterPadContext(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "PadContext", PadContextOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "PadContext", PadContextOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
