//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ADDN_H_
#define MACE_KERNELS_ADDN_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

namespace {
constexpr int kCostPerGroup = 1024;
}  // namespace

template <DeviceType D, typename T>
struct AddNFunctor {
  void operator()(const std::vector<const Tensor *> &input_tensors,
                  Tensor *output_tensor,
                  StatsFuture *future) {
    output_tensor->ResizeLike(input_tensors[0]);
    index_t size = output_tensor->size();
    Tensor::MappingGuard output_map(output_tensor);
    float *output_data = output_tensor->mutable_data<float>();
    memset(output_data, 0, size * sizeof(float));
    int n = input_tensors.size();
    int64_t cost = size * n;
    int64_t groups = 1;
    if (cost > kCostPerGroup) {
      groups = cost / kCostPerGroup;
    }
    int64_t element_per_group = size / groups;

    std::vector<Tensor::MappingGuard> mappers;
    for (int64_t i = 0; i < n; ++i) {
      mappers.emplace_back(Tensor::MappingGuard(input_tensors[i]));
    }

#pragma omp parallel for
    for (int64_t i = 0; i < size; i += element_per_group) {
      int64_t count = std::min(element_per_group, size - i);
      int nn = count >> 2;
      int remain = count - (nn << 2);
      for (int64_t j = 0; j < n; ++j) {
        const float *input_data = input_tensors[j]->data<float>();
        const float *input_ptr = input_data + i;
        float *output_ptr = output_data + i;
        for (int k = 0; k < nn; ++k) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
          float32x4_t in = vld1q_f32(input_ptr);
          float32x4_t out = vld1q_f32(output_ptr);
          out = vaddq_f32(out, in);
          vst1q_f32(output_ptr, out);
#else
          for (int m = 0; m < 4; ++m) {
            output_ptr[m] += input_ptr[m];
          }
#endif

          input_ptr += 4;
          output_ptr += 4;
        }
        for (int k = 0; k < remain; ++k) {
          *output_ptr += *input_ptr;
          ++input_ptr;
          ++output_ptr;
        }
      }
    }
  }
};

template <>
void AddNFunctor<DeviceType::NEON, float>::operator()(
    const std::vector<const Tensor *> &input_tensors,
    Tensor *output_tensor,
    StatsFuture *future);

template <typename T>
struct AddNFunctor<DeviceType::OPENCL, T> {
  void operator()(const std::vector<const Tensor *> &input_tensors,
                  Tensor *output_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ADDN_H_
