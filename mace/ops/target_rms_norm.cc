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

// This op is implemented for kaldi's NormalizeComponent.
// The output y_i = scale * x_i,
// and we want the RMS value of the y_i equals to target_rms,
// so y^t y = Dim * target_rms^2 (if y is one row of the input).
// Dim is the length of a row.
// we need the scale = 1.0 / sqrt(x^t x / (Dim * target_rms^2)).

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class TargetRMSNormOp;

template <typename T>
class TargetRMSNormOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit TargetRMSNormOp(OpConstructContext *context)
      : Operation(context),
        target_rms_(Operation::GetOptionalArg<float>("target_rms", 1.0)),
        add_log_stddev_(
            static_cast<bool>(
                Operation::GetOptionalArg<int>("add_log_stddev", 0))),
        block_dim_(Operation::GetOptionalArg<int>("block_dim", 0)) {}

  // Calculate the square sum of an array
  float SquareSum(const float *data, const index_t data_len) {
    const int num_parts = 4;
    float result = 0.0f;
    if (data_len <= 2 * num_parts) {
      for (index_t i = 0; i < data_len; ++i) {
        result += data[i] * data[i];
      }
    } else {
      const index_t part_len = data_len / num_parts;
      const index_t left_len = data_len % num_parts;
      float results[4] = {0.f, 0.f, 0.f, 0.f};
      for (index_t i = 0; i < num_parts; ++i) {
        for (index_t j = 0; j < part_len; ++j) {
          results[i] += data[i * part_len + j] * data[i * part_len + j];
        }
      }
      for (index_t k = 0; k < left_len; ++k) {
        float d = data[num_parts * part_len + k];
        results[3] += d * d;
      }

      for (index_t i = 0; i < num_parts; ++i) {
        result += results[i];
      }
    }

    return result;
  }


  void NormalizePerRow(const float *data,
                       const index_t data_len,
                       float d_scale,
                       bool add_log_stddev,
                       float *out_data) {
    float scale = SquareSum(data, data_len);
    scale = scale / d_scale;
    scale = scale  < 1.0e-6f ? 1.0e-6f : scale;
    scale = static_cast<float>(1.0 / std::sqrt(scale));
    for (index_t j = 0; j < data_len; ++j) {
      out_data[j] = data[j] * scale;
    }
    if (add_log_stddev) {
      out_data[data_len] = std::log(target_rms_) - std::log(scale);
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();
    const index_t dim_size = input->dim_size();
    MACE_CHECK(dim_size >= 1,
               "TargetRMSNorm's input dim size should be >= 1.");
    const index_t input_dim = input_shape[dim_size -1];
    MACE_CHECK(input_dim > 0 && target_rms_ > 0,
               "Both input dim and target rms should be greater than zero.");
    const index_t bh =
        std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                        std::multiplies<index_t>());
    if (block_dim_ == 0) block_dim_ = static_cast<int>(input_dim);
    const index_t output_dim = add_log_stddev_ ?
                               input_dim + (input_dim / block_dim_) : input_dim;
    std::vector<index_t> output_shape = input->shape();
    output_shape[dim_size - 1] = output_dim;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);

    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    index_t num_rows = bh;
    index_t output_block_dim = add_log_stddev_ ? block_dim_ + 1 : block_dim_;

    if (block_dim_ != input_dim) {
      index_t num_blocks = input_dim / block_dim_;
      num_rows *= num_blocks;
    }
    const float d_scale = block_dim_ * target_rms_ * target_rms_;

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute1D([=](index_t start0, index_t end0, index_t step0) {
      for (index_t i = start0; i < end0; i += step0) {
        const float *input_ptr = input_data + i * block_dim_;
        float *out_ptr = output_data + i * output_block_dim;
        NormalizePerRow(input_ptr,
                        block_dim_,
                        d_scale,
                        add_log_stddev_,
                        out_ptr);
      }
    }, 0, num_rows, 1);


    return MaceStatus::MACE_SUCCESS;
  }

 private:
  float target_rms_;
  bool add_log_stddev_;
  int block_dim_;
};

void RegisterTargetRMSNorm(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "TargetRMSNorm", TargetRMSNormOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
