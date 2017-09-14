//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/addn.h"

namespace mace {
namespace kernels {

template <>
void AddNFunctor<DeviceType::NEON, float>::operator()(const vector<const float*>& inputs,
                                                float *output,
                                                index_t size) {
  // TODO: neon mem copy
  memset(output, 0, size * sizeof(float));
  int n = inputs.size();
  int64_t cost = size * n;
  int64_t groups = 1;
  if (cost > kCostPerGroup) {
    groups = cost / kCostPerGroup;
  }
  int64_t element_per_group = size / groups;

#pragma omp parallel for num_threads(1) // no significant performance improve
  for (int64_t i = 0; i < size; i += element_per_group) {
    int64_t count = std::min(element_per_group, size - i);
    int nn = count >> 2;
    int remain = count - (nn << 2);
    for (int64_t j = 0; j < n; ++j) {
      const float *inptr = inputs[j] + i;
      float *outptr = output + i;
      for (int k = 0; k < nn; ++k) {
        float32x4_t _inptr = vld1q_f32(inptr);
        float32x4_t _outptr = vld1q_f32(outptr);
        _outptr = vaddq_f32(_outptr, _inptr);
        vst1q_f32(outptr, _outptr);

        inptr += 4;
        outptr += 4;
      }
      for (int k = 0; k < remain; ++k) {
        *outptr += *inptr;
        ++inptr;
        ++outptr;
      }
    }
  }
};

} // namespace kernels
} // namespace mace