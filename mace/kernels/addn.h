//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ADDN_H_
#define MACE_KERNELS_ADDN_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

struct AddNFunctorBase {};

template <DeviceType D, typename T>
struct AddNFunctor : AddNFunctorBase {
  void operator()(const std::vector<const Tensor *> &input_tensors,
                  Tensor *output_tensor, StatsFuture *future) {
    output_tensor->ResizeLike(input_tensors[0]);
    Tensor::MappingGuard output_map(output_tensor);
    index_t size = input_tensors[0]->size();
    T *output_ptr = output_tensor->mutable_data<T>();
    memset(output_ptr, 0, size * sizeof(T));
    int n = input_tensors.size();
    for (int i = 0; i < n; ++i) {
      MACE_CHECK(input_tensors[i]->dim(0) == output_tensor->dim(0));
      MACE_CHECK(input_tensors[i]->dim(1) == output_tensor->dim(1));
      MACE_CHECK(input_tensors[i]->dim(2) == output_tensor->dim(2));
      MACE_CHECK(input_tensors[i]->dim(3) == output_tensor->dim(3));
      Tensor::MappingGuard input_map(input_tensors[i]);
      const T *input_ptr = input_tensors[i]->data<T>();
      for (index_t j = 0; j < size; ++j) {
        output_ptr[j] += input_ptr[j];
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
struct AddNFunctor<DeviceType::OPENCL, T> : AddNFunctorBase {
  void operator()(const std::vector<const Tensor *> &input_tensors,
                  Tensor *output_tensor, StatsFuture *future);
};

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_ADDN_H_
