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

// This Op is for IfDefined descriptor in Kaldi.
// It defines time offset.
// If time index <= offset, using zeros as output.
// forward_indexes: indicates which frames will be used for computation.
//                  Because of the model's subsampling, this is pre-computed
//                  in kaldi-onnx.
// cache_forward_indexes: indicates which frames of cached previous output
//                        will be used here. If there is only one input,
//                        this parameter will be empty.

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class IfDefinedOp;

template <typename T>
class IfDefinedOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit IfDefinedOp(OpConstructContext *context)
      : Operation(context),
        forward_indexes_(
            Operation::GetRepeatedArgs<index_t>("forward_indexes")),
        cache_forward_indexes_(
            Operation::GetRepeatedArgs<index_t>("cache_forward_indexes")) {}

  inline void Validate() {
    MACE_CHECK(this->InputSize() <= 2,
               "IfDefined Op should have at most 2 inputs.");
    const Tensor *input = this->Input(INPUT);
    const unsigned int rank = static_cast<unsigned int>(input->dim_size());
    MACE_CHECK(rank >= 2, "IfDefined's input should have at least 2 dims.");
    const index_t input_chunk = input->dim(rank - 2);
    for (size_t i = 0; i < forward_indexes_.size(); ++i) {
      MACE_CHECK(forward_indexes_[i] < input_chunk,
                 "forward index is over range.");
    }
    for (size_t i = 0; i < cache_forward_indexes_.size(); ++i) {
      MACE_CHECK(cache_forward_indexes_[i] < input_chunk &&
                     cache_forward_indexes_[i] >= 0 ,
                 "index is over range.");
    }

    if (this->InputSize() == 2) {
      size_t cache_count = 0;
      for (size_t i = 0; i < forward_indexes_.size(); ++i) {
        if (forward_indexes_[i] < 0)
          cache_count++;
        else
          break;
      }
      MACE_CHECK(cache_forward_indexes_.size() == cache_count,
                 "IfDefined's cache forward index size:",
                 cache_forward_indexes_.size(),
                 " != forward indexes' negative part length:",
                 cache_count);
      for (size_t i = 0; i < cache_forward_indexes_.size(); ++i) {
        MACE_CHECK(cache_forward_indexes_[i] < input_chunk &&
          cache_forward_indexes_[i] >= 0,
          "cache forward index is over range.");
      }
      const Tensor *cache_input = this->Input(CACHE_INPUT);
      MACE_CHECK(cache_input->dim_size() == input->dim_size(),
                 "two inputs should have the same rank");
      for (unsigned int k = 0; k < rank; ++k) {
        MACE_CHECK(input->dim(k) == cache_input->dim(k),
                   "Two inputs should have the same shape");
      }
    }
  }

  void DelayCopy(OpContext *context,
                 const T *input_data,
                 const index_t batch,
                 const index_t chunk,
                 const index_t dim,
                 const std::vector<index_t> &fwd_idxs,
                 T *output_data) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        for (index_t j = start1; j < end1; j += step1) {
          if (fwd_idxs[j] >= 0) {
            memcpy(output_data + (i * chunk + j) * dim,
                   input_data + (i * chunk + fwd_idxs[j]) * dim,
                   dim * sizeof(T));
          }
        }
      }
    }, 0, batch, 1, 0, fwd_idxs.size(), 1);
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    Validate();
    index_t rank = input->dim_size();
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t chunk = input_shape[rank - 2];
    const index_t dim = input_shape[rank - 1];
    std::vector<index_t> output_shape(input->shape());
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    output->Clear();

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    DelayCopy(context,
              input_data,
              batch,
              chunk,
              dim,
              forward_indexes_,
              output_data);

    if (this->InputSize() == 2 && cache_forward_indexes_.size() > 0) {
      const Tensor *cache_input = this->Input(CACHE_INPUT);
      Tensor::MappingGuard cache_input_guard(cache_input);
      const T *cache_input_data = cache_input->data<T>();
      DelayCopy(context,
                cache_input_data,
                batch,
                chunk,
                dim,
                cache_forward_indexes_,
                output_data);
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<index_t> forward_indexes_;
  std::vector<index_t> cache_forward_indexes_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, CACHE_INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterIfDefined(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "IfDefined", IfDefinedOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "IfDefined", IfDefinedOp, DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
