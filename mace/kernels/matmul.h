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
#include <utility>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/gemm.h"
#include "mace/utils/utils.h"
#include "mace/kernels/gemmlowp_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct MatMulFunctor {
  MaceStatus operator()(const Tensor *A,
                        const Tensor *B,
                        Tensor *C,
                        bool transpose_a,
                        bool transpose_b,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    index_t batch;
    index_t height;
    index_t K;
    index_t width;

    index_t rank = A->dim_size();
    height = A->dim(rank - 2);
    K = A->dim(rank - 1);
    if (transpose_a) {
      std::swap(height, K);
    }
    if (transpose_b) {
      width = B->dim(rank - 2);
    } else {
      width = B->dim(rank - 1);
    }
    batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                            std::multiplies<index_t>());

    std::vector<index_t> c_shape = A->shape();
    c_shape[rank - 2] = height;
    c_shape[rank - 1] = width;

    MACE_RETURN_IF_ERROR(C->Resize(c_shape));

    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    const T *a_ptr_base = A->data<T>();
    const T *b_ptr_base = B->data<T>();
    T *c_ptr_base = C->mutable_data<T>();

    // It is better to use large block size if it fits for fast cache.
    // Assume l1 cache size is 32k, we load three blocks at a time (A, B, C),
    // the block size should be sqrt(32k / sizeof(T) / 3).
    memset(c_ptr_base, 0, batch * height * width * sizeof(T));

    Gemm(a_ptr_base, b_ptr_base, batch, height, K, width, c_ptr_base,
         transpose_a, transpose_b);

    return MACE_SUCCESS;
  }
};

template <>
struct MatMulFunctor<CPU, uint8_t> {
  template<gemmlowp::MapOrder AOrder, gemmlowp::MapOrder BOrder>
  void MatMulImpl(const Tensor *A,
                  const Tensor *B,
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  Tensor *C) {
    gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();

    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    auto a_ptr_base = A->data<uint8_t>();
    auto b_ptr_base = B->data<uint8_t>();
    auto c_ptr_base = C->mutable_data<uint8_t>();
    index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                    std::multiplies<index_t>());
    index_t a_size = height * K;
    index_t b_size = K * width;
    index_t c_size = height * width;

    const auto &output_pipeline = GemmlowpOutputPipeline::MakeNoBias(
        A->scale(), B->scale(), C->scale(), C->zero_point());

    for (index_t i = 0; i < batch; ++i) {
      gemmlowp::MatrixMap<const uint8_t, AOrder>
          a_matrix(a_ptr_base + i * a_size, height, K);
      gemmlowp::MatrixMap<const uint8_t, BOrder>
          b_matrix(b_ptr_base + i * b_size, K, width);
      gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::RowMajor>
          c_matrix(c_ptr_base + i * c_size, height, width);

      using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
          &gemm_context, a_matrix, b_matrix, &c_matrix, -A->zero_point(),
          -B->zero_point(), output_pipeline);
    }
  }

  MaceStatus operator()(const Tensor *A,
                        const Tensor *B,
                        Tensor *C,
                        bool transpose_a,
                        bool transpose_b,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    index_t rank = A->dim_size();
    index_t height = A->dim(rank - 2);
    index_t K = A->dim(rank - 1);
    index_t width;

    if (transpose_a) {
      std::swap(height, K);
    }
    if (transpose_b) {
      width = B->dim(rank - 2);
    } else {
      width = B->dim(rank - 1);
    }

    std::vector<index_t> c_shape = A->shape();
    c_shape[rank - 2] = height;
    c_shape[rank - 1] = width;

    MACE_RETURN_IF_ERROR(C->Resize(c_shape));

    constexpr gemmlowp::MapOrder kRowMajor = gemmlowp::MapOrder::RowMajor;
    constexpr gemmlowp::MapOrder kColMajor = gemmlowp::MapOrder::ColMajor;

#define MATMUL_IMPL(AOrder, BOrder) \
    MatMulImpl<AOrder, BOrder>(A, B, height, K, width, C);

    if (transpose_a) {
      if (transpose_b) {
        MATMUL_IMPL(kColMajor, kColMajor);
      } else {
        MATMUL_IMPL(kColMajor, kRowMajor);
      }
    } else {
      if (transpose_b) {
        MATMUL_IMPL(kRowMajor, kColMajor);
      } else {
        MATMUL_IMPL(kRowMajor, kRowMajor);
      }
    }

#undef MATMUL_IMPL

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct MatMulFunctor<DeviceType::GPU, T> {
  MaceStatus operator()(const Tensor *A,
                        const Tensor *B,
                        Tensor *C,
                        bool transpose_a,
                        bool transpose_b,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_MATMUL_H_
