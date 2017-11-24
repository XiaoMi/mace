//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_DSP_UTIL_QUANTIZE_H_
#define MACE_DSP_UTIL_QUANTIZE_H_

#include "mace/core/common.h"
#include "mace/core/tensor.h"

namespace mace {

class Quantizer {
 public:
  Quantizer() {}
  ~Quantizer() {}

  void Quantize(const Tensor &in_tensor,
                Tensor *out_tensor,
                float *min_out, float *max_out);
  void Quantize(const Tensor &in_tensor,
                const float min_in, const float max_in,
                Tensor *out_tensor,
                float *min_out, float *max_out);
  void DeQuantize(const Tensor &in_tensor,
                  const float min_in, const float max_in,
                  Tensor *out_tensor);

 private:
  void QuantizeAdjustRange(float min_in,
                           float max_in,
                           float *min_out,
                           float *max_out,
                           float *stepsize,
                           float *recip_stepsize);

 DISABLE_COPY_AND_ASSIGN(Quantizer);
};

} // mace

#endif // MACE_DSP_UTIL_QUANTIZE_H_
