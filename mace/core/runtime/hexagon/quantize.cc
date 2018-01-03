//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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
  QuantizeAdjustRange(min_in, max_in,
                      min_out, max_out,
                      &stepsize, &recip_stepsize);

  const float *in = in_tensor.data<float>();
  uint8_t *out = out_tensor->mutable_data<uint8_t>();

  for (int i = 0; i < in_tensor.size(); i++) {
    const float inval = in[i];
    float ival = static_cast<uint8_t>((inval - *min_out) * recip_stepsize + 0.5f);
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
  float range = fmaxf(0.0001f, maxval - minval);
  float stepsize = range / 254.0f;
  float recip_stepsize = 254.0f / range;
  // round quantized_zero up so min_out <= minval
  int quantized_zero = ((0.0f - minval) * recip_stepsize) + 0.999;
  float newmin = -quantized_zero * stepsize;
  float newmax = 255.0f * stepsize + newmin;
  *min_out = newmin;
  *max_out = newmax;
  *stepsize_out = stepsize;
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

} // namespace mace