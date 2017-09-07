//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_RELU_H_
#define MACE_KERNELS_RELU_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template<typename T>
void ReluFuntion(const Tensor *input_tensor, Tensor *output_tensor) {
  int64_t size = input_tensor->size();
  output_tensor->ResizeLike(input_tensor);
  const T *input = input_tensor->data<T>();
  T *output = output_tensor->mutable_data<T>();

  for (int64_t i = 0; i < size; ++i) {
    output[i] = std::max(input[i], static_cast<T>(0));
  }
}

} //  namespace kernels
} //  namespace mace

#endif // MACE_KERNELS_RELU_H_