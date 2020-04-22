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

// This Op is for ReplaceIndex in Kaldi.
// Usually used for ivector inputs.
// It copies ivector to each frame of the output.
// forward_indexes: is the pre-computed indexes for output frames.

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class ReplaceIndexOp;

template<typename T>
class ReplaceIndexOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ReplaceIndexOp(OpConstructContext *context)
      : Operation(context),
        forward_indexes_(
            Operation::GetRepeatedArgs<index_t>("forward_indexes")) {}

  inline void Validate() {
    const Tensor *input = this->Input(0);
    const unsigned int rank = static_cast<unsigned int>(input->dim_size());
    MACE_CHECK(rank >= 2, "ReplaceIndex's input should have at least 2 dims.");

    const index_t input_chunk = input->dim(rank - 2);
    for (size_t i = 0; i < forward_indexes_.size(); ++i) {
      MACE_CHECK(forward_indexes_[i] < input_chunk && forward_indexes_[i] >= 0 ,
                 "index is over range.");
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    Validate();
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input->shape().begin(), input->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t rank = input->dim_size();
    const index_t num_ivectors = input_shape[rank - 2];
    const index_t dim = input_shape[rank - 1];
    const index_t input_stride = num_ivectors * dim;

    const index_t out_chunk = forward_indexes_.size();
    const index_t output_stride = out_chunk * dim;

    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 2] = out_chunk;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t i = start1; i < end1; i += step1) {
          memcpy(output_data + b * output_stride + i * dim,
                 input_data + b * input_stride + forward_indexes_[i] * dim,
                 dim * sizeof(T));
        }
      }
    }, 0, batch, 1, 0, out_chunk, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<index_t> forward_indexes_;
};

void RegisterReplaceIndex(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ReplaceIndex", ReplaceIndexOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ReplaceIndex", ReplaceIndexOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
