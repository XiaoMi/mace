//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/neon/relu_neon.h"

namespace mace {
namespace kernels{

void NeonReluFuntion_float(const Tensor *input_tensor,
                           Tensor *output_tensor) {
  int64 size = input_tensor->size();
  output_tensor->ResizeLike(input_tensor);
  const float* input = input_tensor->data<float>();
  float* output = output_tensor->mutable_data<float>();

  float32x4_t _zero = vdupq_n_f32(0.f);
  for (; size > 0; size--) {
    float32x4_t _inp = vld1q_f32(input);
    float32x4_t _outp = vmaxq_f32(_inp, _zero);
    vst1q_f32(output, _outp);

    input += 4;
    output += 4;
  }
}

} // namespace kernels
} // namespace mace