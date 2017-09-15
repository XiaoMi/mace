//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/relu.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
void ReluFunctor<DeviceType::NEON, float>::operator()(const float *input,
                                                      float *output,
                                                      index_t size) {
#pragma omp parallel for num_threads(1)  // no significant performance improve
  for (int64_t i = 0; i < size; i += kCostPerGroup) {
    int64_t count = std::min(static_cast<int64_t>(kCostPerGroup), size - i);
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
};

}  // namespace kernels
}  // namespace mace