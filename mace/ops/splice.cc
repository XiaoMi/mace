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
// (copy and append the frame whose time-index is in context_)
// The context_ values indicate which frame (over time) to splice.
// It will reduce frames because of left context and right context.
// i.e., give input data with shape {20, 40}, and contexts:{-2, -1, 0, 1, 2},
// the output shape should be {16, 200}
// if const_component_dim_ != 0, const_dim_ will be used to determine which
// row of "in" we copy the last part of each row of "out" from (this part is
// not subject to splicing, it's assumed constant for each frame of "input".

#include <functional>
#include <memory>

#include "mace/core/operator.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class SpliceOp;

template<typename T>
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
    MACE_CHECK(input->dim_size() >= 2)
      << "Splice's input's rank should be greater than 2.";

    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();

    const index_t batch =
        std::accumulate(input->shape().begin(), input->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t rank = input->dim_size();
    const index_t chunk = input_shape[rank - 2];
    const index_t input_dim = input_shape[rank - 1];
    const index_t input_stride = chunk * input_dim;

    const index_t num_splice = static_cast<index_t>(context_.size());
    const index_t dim = input_dim - const_dim_;
    const index_t left_context = context_[0];
    const index_t right_context = context_[num_splice -1];

    const index_t out_chunk = chunk - (right_context - left_context);

    MACE_CHECK(input_dim > const_dim_,
               "input dim:", input_dim,
               "should be greater than const dim:", const_dim_);
    const index_t output_dim = dim * num_splice + const_dim_;
    const index_t output_stride = out_chunk * output_dim;

    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 2] = out_chunk;
    output_shape[rank - 1] = output_dim;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    for (int b = 0; b < batch; ++b) {
      for (index_t i = 0; i < out_chunk; ++i) {
        for (index_t c = 0; c < num_splice; ++c) {
          const index_t offset = i + context_[c] - left_context;
          T *output_base =
              output_data + b * output_stride + i * output_dim + c * dim;
          const T *input_base =
              input_data + b * input_stride + offset * input_dim;
          memcpy(output_base, input_base, dim * sizeof(T));
        }
      }
    }

    if (const_dim_ > 0) {
      const index_t output_offset = output_dim - const_dim_;
      const index_t input_offset = dim;
      for (int b = 0; b < batch; ++b) {
        for (index_t i = 0; i < out_chunk; ++i) {
          T *output_base = output_data + b * output_stride + i * output_dim;
          const T *input_base = input_data + b * input_stride + i * input_dim;
          memcpy(output_base + output_offset,
                 input_base + input_offset,
                 const_dim_ * sizeof(T));
        }
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
