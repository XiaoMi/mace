//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/relu.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
void ReluFunctor<DeviceType::NEON, float>::operator()(const Tensor *input_tensor,
                                                      Tensor *output_tensor) {
  const float *input = input_tensor->data<float>();
  float *output = output_tensor->mutable_data<float>();
  index_t size = input_tensor->size();
  if (max_limit_ < 0) {
#pragma omp parallel for
    for (int64_t i = 0; i < size; i += kCostPerGroup) {
      int64_t count = std::min(static_cast<int64_t>(kCostPerGroup), size - i);
      int block = count >> 2;
      int remain = count - (block << 2);
      const float *inptr = input + i;
      float *outptr = output + i;
      float32x4_t zero = vdupq_n_f32(0.f);
      for (; block > 0; --block) {
        float32x4_t in = vld1q_f32(inptr);
        float32x4_t out = vmaxq_f32(in, zero);
        vst1q_f32(outptr, out);

        inptr += 4;
        outptr += 4;
      }
      for (; remain > 0; --remain) {
        *outptr = std::max(*inptr, 0.f);
        ++inptr;
        ++outptr;
      }
    }
  } else {
#pragma omp parallel for
    for (int64_t i = 0; i < size; i += kCostPerGroup) {
      int64_t count = std::min(static_cast<int64_t>(kCostPerGroup), size - i);
      int block = count >> 2;
      int remain = count - (block << 2);
      const float *inptr = input + i;
      float *outptr = output + i;
      float32x4_t zero = vdupq_n_f32(0.f);
      float32x4_t vmax = vdupq_n_f32(max_limit_);
      for (; block > 0; --block) {
        float32x4_t in = vld1q_f32(inptr);
        float32x4_t out = vmaxq_f32(in, zero);
        out = vminq_f32(out, vmax);
        vst1q_f32(outptr, out);

        inptr += 4;
        outptr += 4;
      }
      for (; remain > 0; --remain) {
        *outptr = std::min(std::max(*inptr, 0.f), max_limit_);
        ++inptr;
        ++outptr;
      }
    }
  }
};

}  // namespace kernels
}  // namespace mace