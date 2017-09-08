//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_RELU_H_
#define MACE_KERNELS_RELU_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct ReluFunctor {
  void operator()(const T *input, T *output, index_t size) {
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::max(input[i], static_cast<T>(0));
    }
  }
};

template <>
void ReluFunctor<DeviceType::NEON, float>::operator()(const float *input,
                                                      float *output,
                                                      index_t size);

} //  namespace kernels
} //  namespace mace

#endif // MACE_KERNELS_RELU_H_