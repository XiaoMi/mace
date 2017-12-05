//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BIAS_ADD_H_
#define MACE_KERNELS_BIAS_ADD_H_

#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct BiasAddFunctor {
  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output) {
    const index_t batch = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);

    const T *input_ptr = input->data<T>();
    const T *bias_ptr = bias->data<T>();
    T *output_ptr = output->mutable_data<T>();


    index_t pos = 0;
#pragma omp parallel for
    for (index_t n = 0; n < batch; ++n) {
      for (index_t h = 0; h < height; ++h) {
        for (index_t w = 0; w < width; ++w) {
          for (index_t c = 0; c < channels; ++c) {
            output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
            ++pos;
          }
        }
      }
    }

  }
};

/*
template <>
void BiasAddFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *bias,
    Tensor *output);
*/

template <typename T>
struct BiasAddFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output);
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BIAS_ADD_H_
