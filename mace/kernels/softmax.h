//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_SOFTMAX_H_
#define MACE_KERNELS_SOFTMAX_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct SoftmaxFunctor {
  void operator()(const Tensor *logits, Tensor *output, StatsFuture *future) {
    Tensor::MappingGuard logits_guard(logits);
    Tensor::MappingGuard output_guard(output);
    const T *logits_ptr = logits->data<T>();
    T *output_ptr = output->mutable_data<T>();
    auto &logits_shape = logits->shape();
    const index_t batch_size =
        std::accumulate(logits_shape.begin(), logits_shape.end() - 1, 1,
                        std::multiplies<index_t>());
    const index_t num_classes = logits_shape.back();

#pragma omp parallel
    {
      // Allocate per thread buffer
      std::vector<T> exp_data(num_classes);
#pragma omp for
      for (index_t i = 0; i < batch_size; ++i) {
        const index_t pos = i * num_classes;
        T max_value = logits_ptr[pos];
        for (index_t c = 1; c < num_classes; ++c) {
          max_value = std::max(max_value, logits_ptr[pos + c]);
        }
        // TODO(liuqi): check overflow?
        T sum = 0;
        for (index_t c = 0; c < num_classes; ++c) {
          exp_data[c] = ::exp((logits_ptr[pos + c] - max_value));
          sum += exp_data[c];
        }
        for (index_t c = 0; c < num_classes; ++c) {
          output_ptr[pos + c] = exp_data[c] / sum;
        }
      }
    }
  }
};

template <typename T>
struct SoftmaxFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *logits, Tensor *output, StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SOFTMAX_H_
