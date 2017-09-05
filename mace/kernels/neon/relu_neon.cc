//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/neon/relu_neon.h"

namespace mace {
namespace kernels {

void NeonReluFuntion_float(const Tensor *input_tensor,
                           Tensor *output_tensor) {
  int64 size = input_tensor->size();
  output_tensor->ResizeLike(input_tensor);
  const float *input = input_tensor->data<float>();
  float *output = output_tensor->mutable_data<float>();

#pragma omp parallel for num_threads(1) // no significant performance improve
  for (int64 i = 0; i < size; i += kCostPerGroup) {
    int64 count = std::min(static_cast<int64>(kCostPerGroup), size - i);
    int nn = count >> 2;
    int remain = count - (nn << 2);
    const float *inptr = input + i;
    float *outptr = output + i;
    float32x4_t _zero = vdupq_n_f32(0.f);
    for (; nn > 0; --nn) {
      float32x4_t _inptr = vld1q_f32(inptr);
      float32x4_t _outptr = vmaxq_f32(_inptr, _zero);
      vst1q_f32(outptr, _outptr);

      inptr += 4;
      outptr += 4;
    }
    for (; remain > 0; --remain) {
      *outptr = std::max(*inptr, 0.f);
      ++inptr;
      ++outptr;
    }
  }
}

} // namespace kernels
} // namespace mace