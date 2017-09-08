//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/neon/addn_neon.h"
#include "mace/core/common.h"

namespace mace {
namespace kernels {

void NeonAddNFuntion_float(const vector<const Tensor *> &input_tensor,
                           Tensor *output_tensor) {
  int n = input_tensor.size();
  MACE_CHECK(n > 1);
  MACE_CHECK_NOTNULL(input_tensor[0]);
  int64_t size = input_tensor[0]->size();
  output_tensor->ResizeLike(input_tensor[0]);
  float *output = output_tensor->mutable_data<float>();
  vector<const float *> inputs(n);
  for (int i = 0; i < n; ++i) {
    inputs[i] = input_tensor[i]->data<float>();
  }

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
}

} // namespace kernels
} // namespace mace