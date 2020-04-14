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
// forward_indexes and forward_const_indexes indicate which frames will
// be used for computation, and they are precomputed in kaldi-onnx converter
// becase of supporting subsample.

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
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
        context_(Operation::GetRepeatedArgs<index_t>("context")),
        const_dim_(
            Operation::GetOptionalArg<int>("const_component_dim", 0)),
        forward_indexes_(
            Operation::GetRepeatedArgs<index_t>("forward_indexes")),
        forward_const_indexes_(
            Operation::GetRepeatedArgs<index_t>("forward_const_indexes")) {}

  inline void Validate() {
    MACE_CHECK(context_.size() > 0)
        << "The context param should not be empty in Splice Op.";
    MACE_CHECK(forward_indexes_.size() % context_.size() == 0,
               "Splice's forward indexes should be multiply of num splice.");
    const Tensor *input = this->Input(0);
    const unsigned int rank = static_cast<unsigned int>(input->dim_size());
    MACE_CHECK(rank >= 2, "Splice's input should have at least 2 dims.");
    MACE_CHECK(input->dim(rank - 1) > const_dim_,
               "input dim:", input->dim(rank - 1),
               "should be greater than const dim:", const_dim_);

    const index_t input_chunk = input->dim(rank - 2);
    for (size_t i = 0; i < forward_indexes_.size(); ++i) {
      MACE_CHECK(forward_indexes_[i] < input_chunk && forward_indexes_[i] >= 0)
          << " forward index:" << forward_indexes_[i] << " input shape:"
          << input->dim(0) << "," << input->dim(1) << "," << input->dim(2);
    }
    for (size_t i = 0; i < forward_const_indexes_.size(); ++i) {
      MACE_CHECK(forward_const_indexes_[i] < input_chunk &&
                     forward_const_indexes_[i] >= 0 ,
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
    const index_t chunk = input_shape[rank - 2];
    const index_t input_dim = input_shape[rank - 1];
    const index_t input_stride = chunk * input_dim;

    const index_t num_splice = static_cast<index_t>(context_.size());
    const index_t dim = input_dim - const_dim_;
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    const index_t out_chunk = forward_indexes_.size() / num_splice;
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

    thread_pool.Compute3D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1,
                              index_t start2, index_t end2, index_t step2) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t i = start1; i < end1; i += step1) {
          for (index_t c = start2; c < end2; c += step2) {
            const index_t pos = forward_indexes_[i * num_splice + c];
            T *output_base =
                output_data + b * output_stride + i * output_dim + c * dim;
            const T *input_base =
                input_data + b * input_stride + pos * input_dim;
            memcpy(output_base, input_base, dim * sizeof(T));
          }
        }
      }
    }, 0, batch, 1, 0, out_chunk, 1, 0, num_splice, 1);

    if (const_dim_ > 0) {
      const index_t output_offset = output_dim - const_dim_;
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            T *output_base = output_data + b * output_stride +
                i * output_dim + output_offset;
            const T *input_base =
                input_data + b * input_stride +
                forward_const_indexes_[i] * input_dim + dim;
            memcpy(output_base, input_base,
                   const_dim_ * sizeof(T));
          }
        }
      }, 0, batch, 1, 0, out_chunk, 1);
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<index_t> context_;
  int const_dim_;
  std::vector<index_t> forward_indexes_;
  std::vector<index_t> forward_const_indexes_;
};

void RegisterSplice(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Splice", SpliceOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Splice", SpliceOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
