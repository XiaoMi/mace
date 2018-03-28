//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_NEGATIVE_H_
#define MACE_KERNELS_NEGATIVE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct NegFunctor {
  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    const index_t batch = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);

    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for collapse(4)
    for (index_t n = 0; n < batch; ++n) {
      for (index_t h = 0; h < height; ++h) {
        for (index_t w = 0; w < width; ++w) {
          for (index_t c = 0; c < channels; ++c) {
            index_t pos = (((n * height) + h) * width + w) * channels + c;
            output_ptr[pos] = 0 - input_ptr[pos];
          }
        }
      }
    }
  }
};

/*
template <>
void NegFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future);
*/

template <typename T>
struct NegFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);
  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_NEGATIVE_H_
