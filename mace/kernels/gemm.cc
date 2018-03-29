//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include <math.h>
#include <algorithm>

#include "mace/kernels/gemm.h"
#include "mace/utils/utils.h"
#include "mace/utils/logging.h"

namespace mace {
namespace kernels {

namespace {
void GemmRef(const float *A,
             const float *B,
             const index_t height,
             const index_t K,
             const index_t width,
             float *C) {
  memset(C, 0, sizeof(float) * height * width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * width + j] += A[i * K + k] * B[k * width + j];
      }
    }
  }
}

inline void GemmBlock(const float *A,
                      const float *B,
                      const index_t height,
                      const index_t K,
                      const index_t width,
                      const index_t stride_k,
                      const index_t stride_w,
                      float *C) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * stride_w + j] += A[i * stride_k + k] * B[k * stride_w + j];
      }
    }
  }
}

// TODO(liyin): may need implement 883 since RGB
inline void Gemm884(const float *a_ptr,
                    const float *b_ptr,
                    index_t stride_w,
                    index_t stride_k,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
    a15;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_k);
  a3 = vld1q_f32(a_ptr + 1 * stride_k + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_k);
  a5 = vld1q_f32(a_ptr + 2 * stride_k + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_k);
  a7 = vld1q_f32(a_ptr + 3 * stride_k + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_k);
  a9 = vld1q_f32(a_ptr + 4 * stride_k + 4);
  a10 = vld1q_f32(a_ptr + 5 * stride_k);
  a11 = vld1q_f32(a_ptr + 5 * stride_k + 4);
  a12 = vld1q_f32(a_ptr + 6 * stride_k);
  a13 = vld1q_f32(a_ptr + 6 * stride_k + 4);
  a14 = vld1q_f32(a_ptr + 7 * stride_k);
  a15 = vld1q_f32(a_ptr + 7 * stride_k + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_w);
  b2 = vld1q_f32(b_ptr + 2 * stride_w);
  b3 = vld1q_f32(b_ptr + 3 * stride_w);
  b4 = vld1q_f32(b_ptr + 4 * stride_w);
  b5 = vld1q_f32(b_ptr + 5 * stride_w);
  b6 = vld1q_f32(b_ptr + 6 * stride_w);
  b7 = vld1q_f32(b_ptr + 7 * stride_w);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_w);
  c2 = vld1q_f32(c_ptr + 2 * stride_w);
  c3 = vld1q_f32(c_ptr + 3 * stride_w);
  c4 = vld1q_f32(c_ptr + 4 * stride_w);
  c5 = vld1q_f32(c_ptr + 5 * stride_w);
  c6 = vld1q_f32(c_ptr + 6 * stride_w);
  c7 = vld1q_f32(c_ptr + 7 * stride_w);

#define MACE_CONV_1x1_REG_CAL(RC, RA, RAN) \
  c##RC = vfmaq_laneq_f32(c##RC, b0, a##RA, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b1, a##RA, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b2, a##RA, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b3, a##RA, 3); \
  c##RC = vfmaq_laneq_f32(c##RC, b4, a##RAN, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b5, a##RAN, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b6, a##RAN, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b7, a##RAN, 3);

  MACE_CONV_1x1_REG_CAL(0, 0, 1);
  MACE_CONV_1x1_REG_CAL(1, 2, 3);
  MACE_CONV_1x1_REG_CAL(2, 4, 5);
  MACE_CONV_1x1_REG_CAL(3, 6, 7);
  MACE_CONV_1x1_REG_CAL(4, 8, 9);
  MACE_CONV_1x1_REG_CAL(5, 10, 11);
  MACE_CONV_1x1_REG_CAL(6, 12, 13);
  MACE_CONV_1x1_REG_CAL(7, 14, 15);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_w, c1);
  vst1q_f32(c_ptr + 2 * stride_w, c2);
  vst1q_f32(c_ptr + 3 * stride_w, c3);
  vst1q_f32(c_ptr + 4 * stride_w, c4);
  vst1q_f32(c_ptr + 5 * stride_w, c5);
  vst1q_f32(c_ptr + 6 * stride_w, c6);
  vst1q_f32(c_ptr + 7 * stride_w, c7);

#else
  GemmBlock(a_ptr, b_ptr, 8, 8, 4, stride_k, stride_w, c_ptr);
#endif
}

inline void GemmTile(const float *A,
                     const float *B,
                     const index_t height,
                     const index_t K,
                     const index_t width,
                     const index_t stride_k,
                     const index_t stride_w,
                     float *C) {
  index_t h, w, k;
  for (h = 0; h + 7 < height; h += 8) {
    for (w = 0; w + 3 < width; w += 4) {
      for (k = 0; k + 7 < K; k += 8) {
        const float *a_ptr = A + (h * stride_k + k);
        const float *b_ptr = B + (k * stride_w + w);
        float *c_ptr = C + (h * stride_w + w);
        Gemm884(a_ptr, b_ptr, stride_w, stride_k, c_ptr);
      }
      if (k < K) {
        const float *a_ptr = A + (h * stride_k + k);
        const float *b_ptr = B + (k * stride_w + w);
        float *c_ptr = C + (h * stride_w + w);
        GemmBlock(a_ptr, b_ptr, 8, K - k, 4, stride_k, stride_w, c_ptr);
      }
    }
    if (w < width) {
      const float *a_ptr = A + h * stride_k;
      const float *b_ptr = B + w;
      float *c_ptr = C + (h * stride_w + w);
      GemmBlock(a_ptr,
                b_ptr,
                8,
                K,
                width - w,
                stride_k,
                stride_w,
                c_ptr);
    }
  }
  if (h < height) {
    // TODO(liyin): may use Gemm444
    const float *a_ptr = A + (h * stride_k);
    const float *b_ptr = B;
    float *c_ptr = C + h * stride_w;
    GemmBlock(a_ptr,
              b_ptr,
              height - h,
              K,
              width,
              stride_k,
              stride_w,
              c_ptr);
  }
}
}  // namespace

void Gemm(const float *A,
          const float *B,
          const index_t batch,
          const index_t height,
          const index_t K,
          const index_t width,
          float *C) {
  memset(C, 0, sizeof(float) * batch * height * width);


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

#pragma omp parallel for collapse(3)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t bh = 0; bh < block_tile_height; ++bh) {
      for (index_t bw = 0; bw < block_tile_width; ++bw) {
        const float *a_base = A + n * height * K;
        const float *b_base = B + n * K * width;
        float *c_base = C + n * height * width;

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
          GemmTile(a_base + (ih_begin * K + ik_begin),
                   b_base + (ik_begin * width + iw_begin),
                   ih_end - ih_begin,
                   ik_end - ik_begin,
                   iw_end - iw_begin,
                   K,
                   width,
                   c_base + (ih_begin * width + iw_begin));
        }  // bk
      }  // bw
    }  // bh
  }  // n
}

}  // namespace kernels
}  // namespace mace
