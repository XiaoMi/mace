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

// This Op is for Kaldi's BatchNormComponent
// More details about forward computation are here:
// http://kaldi-asr.org/doc/nnet-normalize-component_8cc_source.html#l00320
#include <memory>
#include <string>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/runtime/runtime.h"

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class KaldiBatchNormOp;

template<class T>
class KaldiBatchNormOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit KaldiBatchNormOp(OpConstructContext *context)
      : Operation(context),
        epsilon_(Operation::GetOptionalArg<float>("epsilon",
                                                  static_cast<float>(1e-3))),
        target_rms_(Operation::GetOptionalArg<float>("target_rms", 1.0f)),
        block_dim_(Operation::GetOptionalArg<int>("block_dim", -1)),
        test_mode_(static_cast<bool>(
                       Operation::GetOptionalArg<int>("test_mode", 0))) {}

  void CalculateMeanVar(const T *input_data,
                        index_t length,
                        index_t stride,
                        float mean_scale,
                        float var_scale,
                        T *mean_data,
                        T *var_data) {
    float mean_value = 0.f;
    float var_value = 0.f;
    for (index_t i = 0; i < length; ++i) {
      float x = input_data[i * stride];
      mean_value += x;
      var_value += x * x;
    }
    mean_value = mean_value * mean_scale;
    var_value = var_value * mean_scale;
    float mean_sqr = mean_value * mean_value;
    var_value = (var_value > mean_sqr) ?
                var_scale * (var_value - mean_sqr + epsilon_) :
                var_scale * epsilon_;
    var_data[0] = std::pow(var_value, -0.5f);
    mean_data[0] = mean_value;
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const std::vector<index_t> &input_shape = input->shape();
    const index_t rank = input->dim_size();
    const index_t dim = input_shape[rank - 1];
    if (block_dim_ == -1) block_dim_ = static_cast<int>(dim);
    MACE_CHECK(target_rms_ > 0 && dim > 0 && dim % block_dim_ == 0);
    MACE_CHECK(rank >= 2, "KaldiBatchNorm's input's rank must >= 2.");
    index_t num_rows =
        std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                        std::multiplies<index_t>());

    const index_t blocks = dim / block_dim_;
    if (blocks > 1) num_rows *= blocks;
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

    if (test_mode_) {
      MACE_CHECK(this->InputSize() == 3, "KaldiBatchNorm should have 3 inputs");
      const Tensor *scale = this->Input(SCALE);
      const Tensor *offset = this->Input(OFFSET);
      MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
                 scale->dim_size());
      MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
                 offset->dim_size());
      MACE_CHECK(scale->size() == offset->size()
                     && scale->size() == block_dim_);

      const T *scale_data = scale->data<T>();
      const T *offset_data = offset->data<T>();

      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            index_t idx = i * block_dim_ + j;
            output_data[idx] = input_data[idx] * scale_data[j] + offset_data[j];
          }
        }
      }, 0, num_rows, 1, 0, block_dim_, 1);
    } else {
      auto *runtime = context->runtime();
      auto data_type = DataTypeToEnum<T>::v();
      auto mem_type = input->memory_type();

      Tensor mean(runtime, data_type, mem_type, {block_dim_});
      runtime->AllocateBufferForTensor(&mean, RENT_SCRATCH);
      T *mean_data = mean.mutable_data<T>();

      Tensor var(runtime, data_type, mem_type, {block_dim_});
      runtime->AllocateBufferForTensor(&var, RENT_SCRATCH);
      T *var_data = var.mutable_data<T>();

      float var_scale = 1.0f / (target_rms_ * target_rms_);
      float mean_scale = 1.0f / num_rows;

      thread_pool.Compute1D([=](index_t start0, index_t end0, index_t step0) {
        for (index_t i = start0; i < end0; i += step0) {
          CalculateMeanVar(input_data + i,
                           num_rows,
                           block_dim_,
                           mean_scale,
                           var_scale,
                           mean_data + i,
                           var_data + i);
        }
      }, 0, block_dim_, 1);
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            index_t idx = i * block_dim_ + j;
            output_data[idx] = (input_data[idx] - mean_data[j]) * var_data[j];
          }
        }
      }, 0, num_rows, 1, 0, block_dim_, 1);
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  const float epsilon_;
  const float target_rms_;
  int block_dim_;
  const bool test_mode_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterKaldiBatchNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "KaldiBatchNorm", KaldiBatchNormOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "KaldiBatchNorm", KaldiBatchNormOp,
                        RuntimeType::RT_CPU);
}

}  // namespace ops
}  // namespace mace
