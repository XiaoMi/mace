//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_SOFTMAX_H_
#define MACE_KERNELS_SOFTMAX_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/public/mace.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct SoftmaxFunctor {
  void operator()(const Tensor *logits,
                  Tensor *output,
                  StatsFuture *future) {

    Tensor::MappingGuard logits_guard(logits);
    Tensor::MappingGuard output_guard(output);
    const T *logits_ptr = logits->data<T>();
    T *output_ptr = output->mutable_data<T>();
    auto &logits_shape = logits->shape();
    const index_t batch_size = std::accumulate(logits_shape.begin(), logits_shape.end()-1,
                                               1, std::multiplies<index_t>());
    const index_t num_classes = logits_shape.back();
#pragma omp parallel for
    for (index_t i = 0; i < batch_size; ++i) {
      T max_value = *logits_ptr;
      for (index_t c = 1; c < num_classes; ++c) {
        max_value = std::max(max_value, logits_ptr[c]);
      }
      // TODO: check overflow?
      T sum = 0;
      std::vector<T> exp_data(num_classes);
      for (index_t c = 0; c < num_classes; ++c) {
        exp_data[c] = ::exp((*logits_ptr - max_value));
        sum += exp_data[c];
        logits_ptr++;
      }
      for (index_t c = 0; c < num_classes; ++c) {
        *output_ptr = exp_data[c] / sum;
        output_ptr++;
      }
    }
  }
};


template<typename T>
struct SoftmaxFunctor<DeviceType::OPENCL, T> {

  void operator()(const Tensor *logits,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_SOFTMAX_H_
