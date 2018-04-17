// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/core/runtime/hexagon/quantize.h"

namespace mace {

void Quantizer::Quantize(const Tensor &in_tensor,
                         Tensor *out_tensor,
                         float *min_out,
                         float *max_out) {
  if (in_tensor.size() == 0) return;
  const float *in_data = in_tensor.data<float>();
  float min_in = in_data[0];
  float max_in = in_data[0];
  for (index_t i = 0; i < in_tensor.size(); ++i) {
    min_in = std::min(min_in, in_data[i]);
    max_in = std::max(max_in, in_data[i]);
  }
  Quantize(in_tensor, min_in, max_in, out_tensor, min_out, max_out);
}

void Quantizer::Quantize(const Tensor &in_tensor,
                         const float min_in,
                         const float max_in,
                         Tensor *out_tensor,
                         float *min_out,
                         float *max_out) {
  float stepsize;
  float recip_stepsize;
  QuantizeAdjustRange(min_in, max_in, min_out, max_out, &stepsize,
                      &recip_stepsize);

  const float *in = in_tensor.data<float>();
  uint8_t *out = out_tensor->mutable_data<uint8_t>();

  for (int i = 0; i < in_tensor.size(); i++) {
    const float inval = in[i];
    float ival =
        static_cast<uint8_t>((inval - *min_out) * recip_stepsize + 0.5f);
    if (ival < 0) ival = 0;
    if (ival > 255) ival = 255;
    out[i] = static_cast<uint8_t>(ival);
  }
}

void Quantizer::QuantizeAdjustRange(float min_in,
                                    float max_in,
                                    float *min_out,
                                    float *max_out,
                                    float *stepsize_out,
                                    float *recip_stepsize_out) {
  float minval = std::min(0.0f, min_in);
  float maxval = std::max(0.0f, max_in);
  float range = std::max(0.0001f, maxval - minval);
  float recip_stepsize = 255.0f / range;
  // make z(q0) integer
  if (minval < 0.0f) {
    float z = -minval * recip_stepsize;
    float zi = floorf(z);
    float zf = z - zi;
    if (zf > 0.0001f && zf < 0.9999f) {
      if (zi > 0.0f && (zi >= 254.0f || (zf - 1.0f) * minval > zf * maxval)) {
        range = -255.0f * minval / zi;
        maxval = minval + range;
      } else {
        range = 255.0f * maxval / (254.0f - zi);
        minval = maxval - range;
      }
      recip_stepsize = 255.0f / range;
    }
  }

  *min_out = minval;
  *max_out = maxval;
  *stepsize_out = range / 255.0f;
  *recip_stepsize_out = recip_stepsize;
}

void Quantizer::DeQuantize(const Tensor &in_tensor,
                           const float min_in,
                           const float max_in,
                           Tensor *out_tensor) {
  float range = std::max(0.0001f, max_in - min_in);
  float stepsize = range / 255.0f;

  const uint8_t *in = in_tensor.data<uint8_t>();
  float *out = out_tensor->mutable_data<float>();

  for (int i = 0; i < out_tensor->size(); ++i) {
    out[i] = (in[i] * stepsize) + min_in;
  }
}


}  // namespace mace
