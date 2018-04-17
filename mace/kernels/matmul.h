// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_KERNELS_MATMUL_H_
#define MACE_KERNELS_MATMUL_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/kernels/gemm.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct MatMulFunctor {
  void operator()(const Tensor *A,
                  const Tensor *B,
                  Tensor *C,
                  StatsFuture *future) {
    std::vector<index_t> c_shape = {A->dim(0), A->dim(1), B->dim(2), 1};
    C->Resize(c_shape);

    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    const T *a_ptr_base = A->data<T>();
    const T *b_ptr_base = B->data<T>();
    T *c_ptr_base = C->mutable_data<T>();

    const index_t batch = C->dim(0);
    const index_t height = C->dim(1);
    const index_t width = C->dim(2);
    const index_t K = A->dim(2);
    // It is better to use large block size if it fits for fast cache.
    // Assume l1 cache size is 32k, we load three blocks at a time (A, B, C),
    // the block size should be sqrt(32k / sizeof(T) / 3).
    const index_t block_size = 48;
    const index_t block_tile_height = RoundUpDiv(height, block_size);
    const index_t block_tile_width = RoundUpDiv(width, block_size);
    const index_t block_tile_k = RoundUpDiv(K, block_size);
    const index_t remain_height = height % block_size;
    const index_t remain_width = width % block_size;
    const index_t remain_k = K % block_size;
    constexpr index_t register_tile_size = 4;
    memset(c_ptr_base, 0, batch * height * width * sizeof(T));

    Gemm(a_ptr_base, b_ptr_base, batch, height, K, width, c_ptr_base);
  }
};

template<typename T>
struct MatMulFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *A,
                  const Tensor *B,
                  Tensor *C,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_MATMUL_H_
