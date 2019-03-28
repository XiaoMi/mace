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
        target_rms_(Operation::GetOptionalArg<float>("target_rms", 1.0)) {}

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

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();
    const index_t dim_size = input->dim_size();
    MACE_CHECK(dim_size >= 1,
               "TargetRMSNorm's input dim size should be >= 1.");
    const index_t dim = input_shape[dim_size -1];
    MACE_CHECK(dim > 0 && target_rms_ > 0,
               "Both input dim and target rms should be greater than zero.");
    const index_t bh =
        std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                        std::multiplies<index_t>());
    const float d_scale = dim * target_rms_ * target_rms_;

    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);

    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < bh; ++i) {
      float scale = SquareSum(input_data + i * dim, dim);
      scale = static_cast<float>(1.0 / std::sqrt(scale / d_scale));
      for (index_t j = 0; j < dim; ++j) {
        output_data[i * dim + j] = input_data[i * dim + j] * scale;
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  float target_rms_;
};

void RegisterTargetRMSNorm(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "TargetRMSNorm", TargetRMSNormOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
