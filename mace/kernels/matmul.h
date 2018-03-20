//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_MATMUL_H_
#define MACE_KERNELS_MATMUL_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <string>
#include <vector>
#include <algorithm>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template<typename T,
  int register_tile_size,
  int h_count,
  int w_count,
  int k_count>
inline void MatMulKernelFunc(const T *A,
                             const T *B,
                             T *C,
                             index_t offset_h,
                             index_t offset_w,
                             index_t offset_k,
                             index_t stride_h,
                             index_t stride_w,
                             index_t stride_k) {
  T a_tmp[register_tile_size][register_tile_size] = {0};
  T b_tmp[register_tile_size][register_tile_size] = {0};
  T c_tmp[register_tile_size][register_tile_size] = {0};

  for (int h = 0; h < h_count; ++h) {
    for (int k = 0; k < k_count; ++k) {
      a_tmp[h][k] = A[(offset_h + h) * stride_k + (offset_k + k)];
    }
  }
  for (int k = 0; k < k_count; ++k) {
    for (int w = 0; w < w_count; ++w) {
      b_tmp[k][w] = B[(offset_k + k) * stride_w + (offset_w + w)];
    }
  }

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
  static_assert(register_tile_size == 4, "register tile size must be 4");
  float32x4_t a_dup;
  float32x4_t b_vec[4] =
    {vld1q_f32(b_tmp[0]), vld1q_f32(b_tmp[1]), vld1q_f32(b_tmp[2]),
     vld1q_f32(b_tmp[3])};
  float32x4_t
    c_vec[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};

  for (int h = 0; h < register_tile_size; ++h) {
    for (int k = 0; k < register_tile_size; ++k) {
      a_dup = vdupq_n_f32(a_tmp[h][k]);
      c_vec[h] = vfmaq_f32(c_vec[h], a_dup, b_vec[k]);
    }
  }

  for (int h = 0; h < register_tile_size; ++h) {
    vst1q_f32(c_tmp[h], c_vec[h]);
  }

#else
  for (int h = 0; h < register_tile_size; ++h) {
    for (int w = 0; w < register_tile_size; ++w) {
      for (int k = 0; k < register_tile_size; ++k) {
        c_tmp[h][w] += a_tmp[h][k] * b_tmp[k][w];
      }
    }
  }
#endif

  for (int h = 0; h < h_count; ++h) {
    for (int w = 0; w < w_count; ++w) {
      C[(offset_h + h) * stride_w + (offset_w + w)] += c_tmp[h][w];
    }
  }
}

#define MACE_DO_MATMUL(HC, WC, KC) \
MatMulKernelFunc<T, register_tile_size, HC, WC, KC>(a_ptr_batch_base, \
                b_ptr_batch_base, \
                c_ptr_batch_base, \
                ih, \
                iw, \
                ik, \
                height, \
                width, \
                K);

#define MACE_CASE_K_MATMUL(HC, WC) \
switch (k_count) { \
  case 1: \
    MACE_DO_MATMUL(HC, WC, 1); \
    break; \
  case 2: \
    MACE_DO_MATMUL(HC, WC, 2); \
    break; \
  case 3: \
    MACE_DO_MATMUL(HC, WC, 3); \
    break; \
  case 4: \
    MACE_DO_MATMUL(HC, WC, 4); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported k tile: " << k_count; \
}

#define MACE_CASE_W_MATMUL(HC) \
switch (w_count) { \
  case 1: \
    MACE_CASE_K_MATMUL(HC, 1); \
    break; \
  case 2: \
    MACE_CASE_K_MATMUL(HC, 2); \
    break; \
  case 3: \
    MACE_CASE_K_MATMUL(HC, 3); \
    break; \
  case 4: \
    MACE_CASE_K_MATMUL(HC, 4); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported w tile: " << w_count; \
}

#define MACE_CASE_H_MATMUL \
switch (h_count) { \
  case 1: \
    MACE_CASE_W_MATMUL(1); \
    break; \
  case 2: \
    MACE_CASE_W_MATMUL(2); \
    break; \
  case 3: \
    MACE_CASE_W_MATMUL(3); \
    break; \
  case 4: \
    MACE_CASE_W_MATMUL(4); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported h tile: " << h_count; \
}

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

#pragma omp parallel for collapse(3)
    for (index_t n = 0; n < batch; ++n) {
      // handle block
      for (index_t bh = 0; bh < block_tile_height; ++bh) {
        for (index_t bw = 0; bw < block_tile_width; ++bw) {
          const T *a_ptr_batch_base = a_ptr_base + n * height * K;
          const T *b_ptr_batch_base = b_ptr_base + n * K * width;
          T *c_ptr_batch_base = c_ptr_base + n * height * width;
          const index_t ih_begin = bh * block_size;
          const index_t ih_end =
            bh * block_size + (bh == block_tile_height - 1 && remain_height > 0
                               ? remain_height : block_size);
          const index_t iw_begin = bw * block_size;
          const index_t iw_end =
            bw * block_size
              + (bw == block_tile_width - 1 && remain_width > 0 ? remain_width
                                                                : block_size);

          for (index_t bk = 0; bk < block_tile_k; ++bk) {
            const index_t ik_begin = bk * block_size;
            const index_t ik_end =
              bk * block_size
                + (bk == block_tile_k - 1 && remain_k > 0 ? remain_k
                                                          : block_size);

            // inside block:
            // calculate C[bh, bw] += A[bh, bk] * B[bk, bw] for one k
            for (index_t ih = ih_begin; ih < ih_end;
                 ih += register_tile_size) {
              for (index_t iw = iw_begin; iw < iw_end;
                   iw += register_tile_size) {
                for (index_t ik = ik_begin; ik < ik_end;
                     ik += register_tile_size) {
                  const int h_count = std::min(register_tile_size, ih_end - ih);
                  const int w_count = std::min(register_tile_size, iw_end - iw);
                  const int k_count = std::min(register_tile_size, ik_end - ik);

                  MACE_CASE_H_MATMUL;
                }  // ik
              }  // iw
            }  // ih
          }  // bk
        }  // bw
      }  // bh
    }  // n
  }
};

template<typename T>
struct MatMulFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *A,
                  const Tensor *B,
                  Tensor *C,
                  StatsFuture *future);

  cl::Kernel kernel_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_MATMUL_H_
