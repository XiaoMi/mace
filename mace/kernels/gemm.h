//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_GEMM_H_
#define MACE_KERNELS_GEMM_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {


template <DeviceType D, typename T>
struct GEMMFunctor {
  void operator()(const Tensor *A,
                  const Tensor *B,
                  Tensor *C,
                  StatsFuture *future) {

    std::vector<index_t> c_shape = {A->dim(0), A->dim(1), 1, B->dim(3)};
    C->Resize(c_shape);
    const index_t N = C->dim(0);
    const index_t height = C->dim(1);
    const index_t width = C->dim(3);
    const index_t K = A->dim(3);
    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    const T *a_ptr_base = A->data<T>();
    const T *b_ptr_base = B->data<T>();
    T *c_ptr = C->mutable_data<T>();
    for (int i = 0; i < N; ++i) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          const T *a_ptr = a_ptr_base + h * K;
          const T *b_ptr = b_ptr_base + w;
          *c_ptr = 0;
          for (int k = 0; k < K; ++k) {
            *c_ptr += *a_ptr * *b_ptr;
            a_ptr++;
            b_ptr += width;
          }
          c_ptr++;
        }
      }
      a_ptr_base += height * K;
      b_ptr_base += K * width;
    }
  }
};


template <typename T>
struct GEMMFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *A,
                  const Tensor *B,
                  Tensor *C,
                  StatsFuture *future);
};

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_GEMM_H_
