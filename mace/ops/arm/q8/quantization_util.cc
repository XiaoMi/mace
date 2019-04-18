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

#include "mace/ops/arm/q8/quantization_util.h"

namespace mace {
namespace ops {

const int32_t *GetBiasData(const Tensor *bias,
                           const float input_scale,
                           const float filter_scale,
                           const index_t channels,
                           std::vector<int32_t> *bias_vec) {
  const int32_t *bias_data = nullptr;
  if (bias == nullptr) {
    bias_vec->resize(channels, 0);
    bias_data = bias_vec->data();
  } else {
    auto original_bias_data = bias->data<int32_t>();
    bool adjust_bias_required =
        fabs(input_scale * filter_scale - bias->scale()) > 1e-6;
    if (!adjust_bias_required) {
      bias_data = original_bias_data;
    } else {
      bias_vec->resize(channels);
      float adjust_scale = bias->scale() / (input_scale * filter_scale);
      for (index_t i = 0; i < channels; ++i) {
        (*bias_vec)[i] = static_cast<int32_t>(
            roundf(original_bias_data[i] * adjust_scale));
      }
      bias_data = bias_vec->data();
    }
  }
  return bias_data;
}
}  // namespace ops
}  // namespace mace
