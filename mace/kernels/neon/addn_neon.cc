//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
void AddNFunctor<DeviceType::NEON, float>::operator()(
    const std::vector<const Tensor *> &input_tensors,
    Tensor *output_tensor,
    StatsFuture *future) {
  // TODO: neon mem copy
  output_tensor->ResizeLike(input_tensors[0]);
  index_t size = output_tensor->size();
  float *output_ptr = output_tensor->mutable_data<float>();
  memset(output_ptr, 0, size * sizeof(float));
  int n = input_tensors.size();
  int64_t cost = size * n;
  int64_t groups = 1;
  if (cost > kCostPerGroup) {
    groups = cost / kCostPerGroup;
  }
  int64_t element_per_group = size / groups;

#pragma omp parallel for
  for (int64_t i = 0; i < size; i += element_per_group) {
    int64_t count = std::min(element_per_group, size - i);
    int nn = count >> 2;
    int remain = count - (nn << 2);
    for (int64_t j = 0; j < n; ++j) {
      const float *input_base = input_tensors[j]->data<float>();
      const float *inptr = input_base + i;
      float *outptr = output_ptr + i;
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

}  // namespace kernels
}  // namespace mace
