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

// This Op is for SpliceComponent in Kaldi.
// It splices a context window of frames together [over time]
// (copy and append the frame whose time-index in in context_)
// The context_ values indicate which frame (over time) to splice.
// if context value is less than the first time-index,
// copy and append the first frame's dada,
// when context value is larger than frame's count,
// copy and append the last frame's data.
// i.e., give input data: [[1, 2, 3], [4, 5, 6]],
// with input-dim = 3, frame count = 2, context = [-1, 0, 1]
// Then, the output should be:
// [1, 2, 3, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 4, 5, 6]
// if const_component_dim_ != 0, const_dim_ will be used to determine which
// row of "in" we copy the last part of each row of "out" from (this part is
// not subject to splicing, it's assumed constant for each frame of "input".

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SpliceOp;

template <typename T>
class SpliceOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SpliceOp(OpConstructContext *context)
      : Operation(context),
        context_(Operation::GetRepeatedArgs<int>("context")),
        const_dim_(
            Operation::GetOptionalArg<int>("const_component_dim", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    MACE_CHECK(context_.size() > 0)
      << "The context param should not be empty in Splice Op.";

    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();

    const index_t frames =
        std::accumulate(input->shape().begin(), input->shape().end() - 1, 1,
                        std::multiplies<index_t>());

    const index_t rank = input->dim_size();
    const index_t input_dim = input_shape[rank - 1];

    const index_t num_splice = static_cast<index_t>(context_.size());
    const index_t dim = input_dim - const_dim_;
    MACE_CHECK(input_dim > const_dim_,
               "input dim should be greater than const dim.");
    const index_t output_dim = dim * num_splice + const_dim_;

    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 1] = output_dim;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t i = 0; i < frames; ++i) {
        for (index_t c = 0; c < num_splice; ++c) {
          const index_t offset =
              Clamp<index_t>(context_[c] + i, 0, frames - 1);
          T *output_base = output_data + i * output_dim + c * dim;
          const T *input_base = input_data + offset * input_dim;
          memcpy(output_base, input_base, dim * sizeof(T));
        }
      }

    if (const_dim_ > 0) {
      const index_t output_offset = output_dim - const_dim_;
      const index_t input_offset = dim;
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < frames; ++i) {
          index_t offset = i + context_[0] >= 0 ? i + context_[0] : 0;
          T *output_base = output_data + i * output_dim;
          const T *input_base = input_data + offset * input_dim;
          memcpy(output_base + output_offset,
                 input_base + input_offset,
                 const_dim_ * sizeof(T));
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> context_;
  int const_dim_;
};

void RegisterSplice(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Splice", SpliceOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
