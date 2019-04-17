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

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class PriorBoxOp : public Operation {
 public:
  explicit PriorBoxOp(OpConstructContext *context)
      : Operation(context),
        min_size_(Operation::GetRepeatedArgs<float>("min_size")),
        max_size_(Operation::GetRepeatedArgs<float>("max_size")),
        aspect_ratio_(Operation::GetRepeatedArgs<float>("aspect_ratio")),
        clip_(Operation::GetOptionalArg<bool>("clip", false)),
        variance_(Operation::GetRepeatedArgs<float>("variance")),
        offset_(Operation::GetOptionalArg<float>("offset", 0.5)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *data = this->Input(DATA);
    Tensor *output = this->Output(OUTPUT);
    const std::vector<index_t> &input_shape = input->shape();
    const std::vector<index_t> &data_shape = data->shape();
    const index_t input_h = input_shape[2];
    const index_t input_w = input_shape[3];
    const index_t image_h = data_shape[2];
    const index_t image_w = data_shape[3];
    float step_h = Operation::GetOptionalArg<float>("step_h", 0);
    float step_w = Operation::GetOptionalArg<float>("step_w", 0);
    if (step_h <= 1e-6 || step_w <= 1e-6) {
      step_h = static_cast<float>(image_h) / static_cast<float>(input_h);
      step_w = static_cast<float>(image_w) / static_cast<float>(input_w);
    }
    const index_t num_min_size = min_size_.size();
    MACE_CHECK(num_min_size > 0, "min_size is required!");
    const index_t num_max_size = max_size_.size();
    const index_t num_aspect_ratio = aspect_ratio_.size();

    index_t num_prior = num_min_size * num_aspect_ratio;
    if (num_max_size > 0) {
      MACE_CHECK(max_size_.size() == min_size_.size());
      for (size_t i = 0; i < max_size_.size(); ++i) {
        MACE_CHECK(max_size_[i] > min_size_[i],
                   "max_size must be greater than min_size.");
        num_prior += 1;
      }
    }

    index_t dim = 4 * input_w * input_h * num_prior;
    std::vector<index_t> output_shape = {1, 2, dim};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard output_guard(output);
    T *output_data = output->mutable_data<T>();
    float box_w, box_h;
    for (index_t i = 0; i < input_h; ++i) {
      index_t idx = i * input_w * num_prior * 4;
      for (index_t j = 0; j < input_w; ++j) {
        float center_y = (offset_ + i) * step_h;
        float center_x = (offset_ + j) * step_w;
        for (index_t k = 0; k < num_min_size; ++k) {
          float min_s = min_size_[k];
          box_w = box_h = min_s * 0.5f;
          output_data[idx + 0] = (center_x - box_w) / image_w;
          output_data[idx + 1] = (center_y - box_h) / image_h;
          output_data[idx + 2] = (center_x + box_w) / image_w;
          output_data[idx + 3] = (center_y + box_h) / image_h;
          idx += 4;
          if (num_max_size > 0) {
            float max_s_ = max_size_[k];
            box_w = box_h = sqrt(max_s_ * min_s) * 0.5f;
            output_data[idx + 0] = (center_x - box_w) / image_w;
            output_data[idx + 1] = (center_y - box_h) / image_h;
            output_data[idx + 2] = (center_x + box_w) / image_w;
            output_data[idx + 3] = (center_y + box_h) / image_h;
            idx += 4;
          }
          for (int l = 0; l < num_aspect_ratio; ++l) {
            float ar = aspect_ratio_[l];
            if (fabsf(ar - 1.f) < 1e-6) {
              continue;
            }
            box_w = min_s * sqrt(ar) * 0.5f;
            box_h = min_s / sqrt(ar) * 0.5f;
            output_data[idx + 0] = (center_x - box_w) / image_w;
            output_data[idx + 1] = (center_y - box_h) / image_h;
            output_data[idx + 2] = (center_x + box_w) / image_w;
            output_data[idx + 3] = (center_y + box_h) / image_h;
            idx += 4;
          }
        }
      }
    }

    if (clip_) {
      for (int i = 0; i < dim; ++i) {
        T min = 0;
        T max = 1;
        output_data[i] = std::min(std::max(output_data[i], min), max);
      }
    }

    output_data += dim;
    for (int i = 0; i < dim / 4; ++i) {
      int index = i * 4;
      output_data[0 + index] = variance_[0];
      output_data[1 + index] = variance_[1];
      output_data[2 + index] = variance_[2];
      output_data[3 + index] = variance_[3];
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<float> min_size_;
  std::vector<float> max_size_;
  std::vector<float> aspect_ratio_;
  bool clip_;
  std::vector<float> variance_;
  const float offset_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, DATA);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterPriorBox(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "PriorBox", PriorBoxOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace

