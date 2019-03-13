// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/utils/macros.h"
#include "mace/ops/arm/deconv_2d_neon.h"

namespace mace {
namespace ops {

void Deconv2dNeonK2x2S1(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output) {
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];

  const index_t out_img_size = outh * outw;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t oc = 0; oc < outch; oc += 2) {
      if (oc + 1 < outch) {
        float *out_base0 = output + (b * outch + oc) * out_img_size;
        float *out_base1 = out_base0 + out_img_size;
        for (index_t ic = 0; ic < inch; ++ic) {
          const float *input_base = input + (b * inch + ic) * h * w;
          const float *kernel_base0 = filter + (oc * inch + ic) * 4;
          const float *kernel_base1 = kernel_base0 + inch * 4;
          const float *in = input_base;
          // output channel 0
          const float *k0 = kernel_base0;
          // output channel 1
          const float *k1 = kernel_base1;
#if defined(MACE_ENABLE_NEON)
          // load filter
          float32x4_t k0_vec = vld1q_f32(k0);
          float32x4_t k1_vec = vld1q_f32(k1);
#endif
          for (index_t i = 0; i < h; ++i) {
            float *out_row_base0 = out_base0 + i * outw;
            float *out_row0_0 = out_row_base0;
            float *out_row0_1 = out_row_base0 + outw;

            float *out_row_base1 = out_base1 + i * outw;
            float *out_row1_0 = out_row_base1;
            float *out_row1_1 = out_row_base1 + outw;

            index_t j = 0;
#if defined(MACE_ENABLE_NEON)
            for (; j + 3 < w; j += 4) {
              float32x4_t in_vec = vld1q_f32(in);

              float32x4_t out00, out01, out02, out03;
              float32x4_t out10, out11, out12, out13;

              out00 = vld1q_f32(out_row0_0);
              out00 = neon_vfma_lane_0(out00, in_vec, k0_vec);
              vst1q_f32(out_row0_0, out00);

              out01 = vld1q_f32(out_row0_0 + 1);
              out01 = neon_vfma_lane_1(out01, in_vec, k0_vec);
              vst1q_f32(out_row0_0 + 1, out01);

              out02 = vld1q_f32(out_row0_1);
              out02 = neon_vfma_lane_2(out02, in_vec, k0_vec);
              vst1q_f32(out_row0_1, out02);

              out03 = vld1q_f32(out_row0_1 + 1);
              out03 = neon_vfma_lane_3(out03, in_vec, k0_vec);
              vst1q_f32(out_row0_1 + 1, out03);

              out10 = vld1q_f32(out_row1_0);
              out10 = neon_vfma_lane_0(out10, in_vec, k1_vec);
              vst1q_f32(out_row1_0, out10);

              out11 = vld1q_f32(out_row1_0 + 1);
              out11 = neon_vfma_lane_1(out11, in_vec, k1_vec);
              vst1q_f32(out_row1_0 + 1, out11);

              out12 = vld1q_f32(out_row1_1);
              out12 = neon_vfma_lane_2(out12, in_vec, k1_vec);
              vst1q_f32(out_row1_1, out12);

              out13 = vld1q_f32(out_row1_1 + 1);
              out13 = neon_vfma_lane_3(out13, in_vec, k1_vec);
              vst1q_f32(out_row1_1 + 1, out13);

              in += 4;
              out_row0_0 += 4;
              out_row0_1 += 4;
              out_row1_0 += 4;
              out_row1_1 += 4;
            }
#endif
            for (; j < w; ++j) {
              float val = in[0];
              for (int k = 0; k < 2; ++k) {
                out_row0_0[k] += val * k0[k];
                out_row0_1[k] += val * k0[k + 2];
                out_row1_0[k] += val * k1[k];
                out_row1_1[k] += val * k1[k + 2];
              }
              in++;
              out_row0_0++;
              out_row0_1++;
              out_row1_0++;
              out_row1_1++;
            }
          }
        }
      } else {
        float *out_base0 = output + (b * outch + oc) * outh * outw;
        for (index_t ic = 0; ic < inch; ++ic) {
          const float *input_base = input + (b * inch + ic) * h * w;
          const float *kernel_base0 = filter + (oc * inch + ic) * 4;
          const float *in = input_base;
          const float *k0 = kernel_base0;

#if defined(MACE_ENABLE_NEON)
          // load filter
          float32x4_t k0_vec = vld1q_f32(k0);
#endif
          for (index_t i = 0; i < h; ++i) {
            float *out_row_base0 = out_base0 + i * outw;
            float *out_row0_0 = out_row_base0;
            float *out_row0_1 = out_row_base0 + outw;
            index_t j = 0;
#if defined(MACE_ENABLE_NEON)
            for (; j + 3 < w; j += 4) {
              float32x4_t in_vec = vld1q_f32(in);
              float32x4_t out00, out01, out02, out03;

              out00 = vld1q_f32(out_row0_0);
              out00 = neon_vfma_lane_0(out00, in_vec, k0_vec);
              vst1q_f32(out_row0_0, out00);

              out01 = vld1q_f32(out_row0_0 + 1);
              out01 = neon_vfma_lane_1(out01, in_vec, k0_vec);
              vst1q_f32(out_row0_0 + 1, out01);

              out02 = vld1q_f32(out_row0_1);
              out02 = neon_vfma_lane_2(out02, in_vec, k0_vec);
              vst1q_f32(out_row0_1, out02);

              out03 = vld1q_f32(out_row0_1 + 1);
              out03 = neon_vfma_lane_3(out03, in_vec, k0_vec);
              vst1q_f32(out_row0_1 + 1, out03);

              in += 4;
              out_row0_0 += 4;
              out_row0_1 += 4;
            }
#endif
            for (; j < w; ++j) {
              float val = in[0];
              for (int k = 0; k < 2; ++k) {
                out_row0_0[k] += val * k0[k];
                out_row0_1[k] += val * k0[k + 2];
              }
              in++;
              out_row0_0++;
              out_row0_1++;
            }
          }
        }
      }
    }
  }
}

void Deconv2dNeonK2x2S2(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output) {
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];
  const index_t out_img_size = outh * outw;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t oc = 0; oc < outch; ++oc) {
      float *out_base = output + (b * outch + oc) * out_img_size;
      for (index_t ic = 0; ic < inch; ++ic) {
        const float *input_base = input + (b * inch + ic) * h * w;
        const float *kernel_base = filter + (oc * inch + ic) * 4;
        const float *in = input_base;
        const float *k0 = kernel_base;
#if defined(MACE_ENABLE_NEON)
        float32x4_t k0_vec = vld1q_f32(k0);
#endif
        for (index_t i = 0; i < h; ++i) {
          float *out_row_base = out_base + i * 2 * outw;
          float *out_row_0 = out_row_base;
          float *out_row_1 = out_row_0 + outw;

          index_t j = 0;
#if defined(MACE_ENABLE_NEON)
          for (; j + 3 < w; j += 4) {
            float32x4_t in_vec = vld1q_f32(in);

            // out row 0
            float32x4x2_t out00 = vld2q_f32(out_row_0);
            out00.val[0] =
              neon_vfma_lane_0(out00.val[0], in_vec, k0_vec);
            out00.val[1] =
              neon_vfma_lane_1(out00.val[1], in_vec, k0_vec);
            vst2q_f32(out_row_0, out00);

            // out row 1
            float32x4x2_t out10 = vld2q_f32(out_row_1);
            out10.val[0] =
              neon_vfma_lane_2(out10.val[0], in_vec, k0_vec);
            out10.val[1] =
              neon_vfma_lane_3(out10.val[1], in_vec, k0_vec);
            vst2q_f32(out_row_1, out10);

            in += 4;
            out_row_0 += 8;
            out_row_1 += 8;
          }
#endif
          for (; j < w; ++j) {
            float val = in[0];
            for (int k = 0; k < 2; ++k) {
              out_row_0[k] += val * k0[k];
              out_row_1[k] += val * k0[k + 2];
            }
            in++;
            out_row_0 += 2;
            out_row_1 += 2;
          }
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace mace
