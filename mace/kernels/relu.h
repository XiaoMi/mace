//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_RELU_H_
#define MACE_KERNELS_RELU_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReluFunctor {
  T max_limit_;

  void operator()(const Tensor *input, Tensor *output) {
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();
    index_t size = input->size();
    if (max_limit_ < 0) {
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::max(input_ptr[i], static_cast<T>(0));
      }
    } else {
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::min(std::max(input_ptr[i], static_cast<T>(0)), max_limit_);
      }
    }
  }
};

template <>
void ReluFunctor<DeviceType::NEON, float>::operator()(const Tensor *input,
                                                      Tensor *output);
template <>
void ReluFunctor<DeviceType::OPENCL, float>::operator()(const Tensor *input,
                                                        Tensor *output);

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_RELU_H_