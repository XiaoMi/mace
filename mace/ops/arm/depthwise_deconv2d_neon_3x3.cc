// Copyright 2019 The MACE Authors. All Rights Reserved.
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
#include "mace/ops/arm/depthwise_deconv2d_neon.h"

namespace mace {
namespace ops {

void DepthwiseDeconv2dNeonK3x3S1(const float *input,
                                 const float *filter,
                                 const index_t *in_shape,
                                 const index_t *out_shape,
                                 float *output) {
  const index_t channels = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];
  const index_t in_img_size = h * w;

  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];
  const index_t out_img_size = outh * outw;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t offset = b * channels + c;
      float *out_base = output + offset * out_img_size;
      const float *input_base = input + offset * in_img_size;
      const float *kernel_base = filter + c * 9;
      const float *in = input_base;
      const float *k0 = kernel_base;
      const float *k1 = kernel_base + 3;
      const float *k2 = kernel_base + 5;

#if defined(MACE_ENABLE_NEON)
      // load filter
      float32x4_t k0_vec = vld1q_f32(k0);
      float32x4_t k1_vec = vld1q_f32(k1);
      float32x4_t k2_vec = vld1q_f32(k2);
#endif
      for (index_t i = 0; i < h; ++i) {
        float *out_row_base = out_base + i * outw;
        float *out_row0 = out_row_base;
        float *out_row1 = out_row_base + outw;
        float *out_row2 = out_row_base + 2 * outw;
        index_t j = 0;
#if defined(MACE_ENABLE_NEON)
        for (; j + 3 < w; j += 4) {
          float32x4_t in_vec = vld1q_f32(in);

          float32x4_t out00, out01, out02;
          float32x4_t out10, out11, out12;
          float32x4_t out20, out21, out22;

          out00 = vld1q_f32(out_row0 + 0);
          out00 = neon_vfma_lane_0(out00, in_vec, k0_vec);
          vst1q_f32(out_row0 + 0, out00);

          out01 = vld1q_f32(out_row0 + 1);
          out01 = neon_vfma_lane_1(out01, in_vec, k0_vec);
          vst1q_f32(out_row0 + 1, out01);

          out02 = vld1q_f32(out_row0 + 2);
          out02 = neon_vfma_lane_2(out02, in_vec, k0_vec);
          vst1q_f32(out_row0 + 2, out02);

          out10 = vld1q_f32(out_row1 + 0);
          out10 = neon_vfma_lane_0(out10, in_vec, k1_vec);
          vst1q_f32(out_row1 + 0, out10);

          out11 = vld1q_f32(out_row1 + 1);
          out11 = neon_vfma_lane_1(out11, in_vec, k1_vec);
          vst1q_f32(out_row1 + 1, out11);

          out12 = vld1q_f32(out_row1 + 2);
          out12 = neon_vfma_lane_2(out12, in_vec, k1_vec);
          vst1q_f32(out_row1 + 2, out12);

          out20 = vld1q_f32(out_row2 + 0);
          out20 = neon_vfma_lane_1(out20, in_vec, k2_vec);
          vst1q_f32(out_row2 + 0, out20);

          out21 = vld1q_f32(out_row2 + 1);
          out21 = neon_vfma_lane_2(out21, in_vec, k2_vec);
          vst1q_f32(out_row2 + 1, out21);

          out22 = vld1q_f32(out_row2 + 2);
          out22 = neon_vfma_lane_3(out22, in_vec, k2_vec);
          vst1q_f32(out_row2 + 2, out22);

          in += 4;
          out_row0 += 4;
          out_row1 += 4;
          out_row2 += 4;
        }
#endif
        for (; j < w; ++j) {
          float val = in[0];
          for (int k = 0; k < 3; ++k) {
            out_row0[k] += val * k0[k];
            out_row1[k] += val * k1[k];
            out_row2[k] += val * k2[k + 1];
          }
          in++;
          out_row0++;
          out_row1++;
          out_row2++;
        }
      }
    }
  }
}

void DepthwiseDeconv2dNeonK3x3S2(const float *input,
                                 const float *filter,
                                 const index_t *in_shape,
                                 const index_t *out_shape,
                                 float *output) {
  const index_t channels = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];
  const index_t in_img_size = h * w;

  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];
  const index_t out_img_size = outh * outw;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t offset = b * channels + c;
      float *out_base = output + offset * out_img_size;
      const float *input_base = input + offset * in_img_size;
      const float *kernel_base = filter + c * 9;
      const float *in = input_base;

      const float *k0 = kernel_base;
      const float *k1 = kernel_base + 3;
      const float *k2 = kernel_base + 5;

#if defined(MACE_ENABLE_NEON)
      float32x4_t k0_vec = vld1q_f32(k0);
      float32x4_t k1_vec = vld1q_f32(k1);
      float32x4_t k2_vec = vld1q_f32(k2);
#endif
      for (index_t i = 0; i < h; ++i) {
        float *out_row_base = out_base + i * 2 * outw;
        float *out_row_0 = out_row_base;
        float *out_row_1 = out_row_0 + outw;
        float *out_row_2 = out_row_1 + outw;

        index_t j = 0;
#if defined(MACE_ENABLE_NEON)
        for (index_t n = 0; n + 9 < outw; n += 8) {
          float32x4_t in_vec = vld1q_f32(in);

          // out row 0
          float32x4x2_t out00 = vld2q_f32(out_row_0);
          out00.val[0] =
            neon_vfma_lane_0(out00.val[0], in_vec, k0_vec);
          out00.val[1] =
            neon_vfma_lane_1(out00.val[1], in_vec, k0_vec);
          vst2q_f32(out_row_0, out00);

          float32x4x2_t out01 = vld2q_f32(out_row_0 + 2);
          out01.val[0] =
            neon_vfma_lane_2(out01.val[0], in_vec, k0_vec);
          vst2q_f32(out_row_0 + 2, out01);

          // out row 1
          float32x4x2_t out10 = vld2q_f32(out_row_1);
          out10.val[0] =
            neon_vfma_lane_0(out10.val[0], in_vec, k1_vec);
          out10.val[1] =
            neon_vfma_lane_1(out10.val[1], in_vec, k1_vec);
          vst2q_f32(out_row_1, out10);

          float32x4x2_t out11 = vld2q_f32(out_row_1 + 2);
          out11.val[0] =
            neon_vfma_lane_2(out11.val[0], in_vec, k1_vec);
          vst2q_f32(out_row_1 + 2, out11);

          // out row 2
          float32x4x2_t out20 = vld2q_f32(out_row_2);
          out20.val[0] =
            neon_vfma_lane_1(out20.val[0], in_vec, k2_vec);
          out20.val[1] =
            neon_vfma_lane_2(out20.val[1], in_vec, k2_vec);
          vst2q_f32(out_row_2, out20);

          float32x4x2_t out21 = vld2q_f32(out_row_2 + 2);
          out21.val[0] =
            neon_vfma_lane_3(out21.val[0], in_vec, k2_vec);
          vst2q_f32(out_row_2 + 2, out21);

          in += 4;
          out_row_0 += 8;
          out_row_1 += 8;
          out_row_2 += 8;
          j += 4;
        }
#endif
        for (; j < w; ++j) {
          float val = in[0];

          for (int k = 0; k < 3; ++k) {
            out_row_0[k] += val * k0[k];
            out_row_1[k] += val * k1[k];
            out_row_2[k] += val * k2[k + 1];
          }

          in++;
          out_row_0 += 2;
          out_row_1 += 2;
          out_row_2 += 2;
        }
      }
    }
  }
}

void GroupDeconv2dNeonK3x3S1(const float *input,
                             const float *filter,
                             const int group,
                             const index_t *in_shape,
                             const index_t *out_shape,
                             float *output) {
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];

  const index_t in_img_size = h * w;
  const index_t out_img_size = outh * outw;

  const index_t inch_g = inch / group;
  const index_t outch_g = outch / group;

#pragma omp parallel for collapse(3) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (int g = 0; g < group; ++g) {
      for (index_t oc = 0; oc < outch_g; oc += 2) {
        if (oc + 1 < outch_g) {
          const index_t out_offset = b * outch + outch_g * g + oc;
          float *out_base0 = output + out_offset * out_img_size;
          float *out_base1 = out_base0 + out_img_size;
          for (index_t ic = 0; ic < inch_g; ++ic) {
            const index_t in_offset = b * inch + inch_g * g + ic;
            const float *input_base = input + in_offset * in_img_size;
            const index_t kernel_offset = (oc * group + g) * inch_g + ic;
            const float *kernel_base0 = filter + kernel_offset * 9;
            const float *kernel_base1 = kernel_base0 + inch * 9;
            const float *in = input_base;

            // output channel 0
            const float *k0_0 = kernel_base0;
            const float *k0_1 = kernel_base0 + 3;
            const float *k0_2 = kernel_base0 + 5;
            // output channel 1
            const float *k1_0 = kernel_base1;
            const float *k1_1 = kernel_base1 + 3;
            const float *k1_2 = kernel_base1 + 5;

#if defined(MACE_ENABLE_NEON)
            // load filter
          float32x4_t k00_vec, k01_vec, k02_vec;
          float32x4_t k10_vec, k11_vec, k12_vec;

          k00_vec = vld1q_f32(k0_0);
          k01_vec = vld1q_f32(k0_1);
          k02_vec = vld1q_f32(k0_2);

          k10_vec = vld1q_f32(k1_0);
          k11_vec = vld1q_f32(k1_1);
          k12_vec = vld1q_f32(k1_2);
#endif
            for (index_t i = 0; i < h; ++i) {
              float *out_row_base0 = out_base0 + i * outw;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + outw;
              float *out_row0_2 = out_row_base0 + 2 * outw;

              float *out_row_base1 = out_base1 + i * outw;
              float *out_row1_0 = out_row_base1;
              float *out_row1_1 = out_row_base1 + outw;
              float *out_row1_2 = out_row_base1 + 2 * outw;

              index_t j = 0;
#if defined(MACE_ENABLE_NEON)
              for (; j + 3 < w; j += 4) {
              float32x4_t in_vec = vld1q_f32(in);

              float32x4_t out00, out01, out02;
              float32x4_t out10, out11, out12;
              float32x4_t out20, out21, out22;

              out00 = vld1q_f32(out_row0_0);
              out00 = neon_vfma_lane_0(out00, in_vec, k00_vec);
              vst1q_f32(out_row0_0, out00);

              out01 = vld1q_f32(out_row0_0 + 1);
              out01 = neon_vfma_lane_1(out01, in_vec, k00_vec);
              vst1q_f32(out_row0_0 + 1, out01);

              out02 = vld1q_f32(out_row0_0 + 2);
              out02 = neon_vfma_lane_2(out02, in_vec, k00_vec);
              vst1q_f32(out_row0_0 + 2, out02);

              out10 = vld1q_f32(out_row0_1 + 0);
              out10 = neon_vfma_lane_0(out10, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 0, out10);

              out11 = vld1q_f32(out_row0_1 + 1);
              out11 = neon_vfma_lane_1(out11, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 1, out11);

              out12 = vld1q_f32(out_row0_1 + 2);
              out12 = neon_vfma_lane_2(out12, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 2, out12);

              out20 = vld1q_f32(out_row0_2 + 0);
              out20 = neon_vfma_lane_1(out20, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 0, out20);

              out21 = vld1q_f32(out_row0_2 + 1);
              out21 = neon_vfma_lane_2(out21, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 1, out21);

              out22 = vld1q_f32(out_row0_2 + 2);
              out22 = neon_vfma_lane_3(out22, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 2, out22);

              out00 = vld1q_f32(out_row1_0 + 0);
              out00 = neon_vfma_lane_0(out00, in_vec, k10_vec);
              vst1q_f32(out_row1_0 + 0, out00);

              out01 = vld1q_f32(out_row1_0 + 1);
              out01 = neon_vfma_lane_1(out01, in_vec, k10_vec);
              vst1q_f32(out_row1_0 + 1, out01);

              out02 = vld1q_f32(out_row1_0 + 2);
              out02 = neon_vfma_lane_2(out02, in_vec, k10_vec);
              vst1q_f32(out_row1_0 + 2, out02);

              out10 = vld1q_f32(out_row1_1 + 0);
              out10 = neon_vfma_lane_0(out10, in_vec, k11_vec);
              vst1q_f32(out_row1_1 + 0, out10);

              out11 = vld1q_f32(out_row1_1 + 1);
              out11 = neon_vfma_lane_1(out11, in_vec, k11_vec);
              vst1q_f32(out_row1_1 + 1, out11);

              out12 = vld1q_f32(out_row1_1 + 2);
              out12 = neon_vfma_lane_2(out12, in_vec, k11_vec);
              vst1q_f32(out_row1_1 + 2, out12);

              out20 = vld1q_f32(out_row1_2 + 0);
              out20 = neon_vfma_lane_1(out20, in_vec, k12_vec);
              vst1q_f32(out_row1_2 + 0, out20);

              out21 = vld1q_f32(out_row1_2 + 1);
              out21 = neon_vfma_lane_2(out21, in_vec, k12_vec);
              vst1q_f32(out_row1_2 + 1, out21);

              out22 = vld1q_f32(out_row1_2 + 2);
              out22 = neon_vfma_lane_3(out22, in_vec, k12_vec);
              vst1q_f32(out_row1_2 + 2, out22);

              in += 4;
              out_row0_0 += 4;
              out_row0_1 += 4;
              out_row0_2 += 4;
              out_row1_0 += 4;
              out_row1_1 += 4;
              out_row1_2 += 4;
            }
#endif
              for (; j < w; ++j) {
                float val = in[0];
                for (int k = 0; k < 3; ++k) {
                  out_row0_0[k] += val * k0_0[k];
                  out_row0_1[k] += val * k0_1[k];
                  out_row0_2[k] += val * k0_2[k + 1];
                  out_row1_0[k] += val * k1_0[k];
                  out_row1_1[k] += val * k1_1[k];
                  out_row1_2[k] += val * k1_2[k + 1];
                }
                in++;
                out_row0_0++;
                out_row0_1++;
                out_row0_2++;
                out_row1_0++;
                out_row1_1++;
                out_row1_2++;
              }
            }
          }
        } else {
          const index_t out_offset = b * outch + outch_g * g + oc;
          float *out_base0 = output + out_offset * out_img_size;
          for (index_t ic = 0; ic < inch_g; ++ic) {
            const index_t in_offset = (b * group + g) * inch_g + ic;
            const float *input_base = input + in_offset * in_img_size;
            const index_t kernel_offset = (oc * group + g) * inch_g + ic;
            const float *kernel_base0 = filter + kernel_offset * 9;
            const float *in = input_base;
            const float *k0_0 = kernel_base0;
            const float *k0_1 = kernel_base0 + 3;
            const float *k0_2 = kernel_base0 + 5;

#if defined(MACE_ENABLE_NEON)
            // load filter
          float32x4_t k00_vec = vld1q_f32(k0_0);
          float32x4_t k01_vec = vld1q_f32(k0_1);
          float32x4_t k02_vec = vld1q_f32(k0_2);
#endif
            for (index_t i = 0; i < h; ++i) {
              float *out_row_base0 = out_base0 + i * outw;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + outw;
              float *out_row0_2 = out_row_base0 + 2 * outw;
              index_t j = 0;
#if defined(MACE_ENABLE_NEON)
              for (; j + 3 < w; j += 4) {
              float32x4_t in_vec = vld1q_f32(in);

              float32x4_t out00, out01, out02;
              float32x4_t out10, out11, out12;
              float32x4_t out20, out21, out22;

              out00 = vld1q_f32(out_row0_0 + 0);
              out00 = neon_vfma_lane_0(out00, in_vec, k00_vec);
              vst1q_f32(out_row0_0 + 0, out00);

              out01 = vld1q_f32(out_row0_0 + 1);
              out01 = neon_vfma_lane_1(out01, in_vec, k00_vec);
              vst1q_f32(out_row0_0 + 1, out01);

              out02 = vld1q_f32(out_row0_0 + 2);
              out02 = neon_vfma_lane_2(out02, in_vec, k00_vec);
              vst1q_f32(out_row0_0 + 2, out02);

              out10 = vld1q_f32(out_row0_1 + 0);
              out10 = neon_vfma_lane_0(out10, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 0, out10);

              out11 = vld1q_f32(out_row0_1 + 1);
              out11 = neon_vfma_lane_1(out11, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 1, out11);

              out12 = vld1q_f32(out_row0_1 + 2);
              out12 = neon_vfma_lane_2(out12, in_vec, k01_vec);
              vst1q_f32(out_row0_1 + 2, out12);

              out20 = vld1q_f32(out_row0_2 + 0);
              out20 = neon_vfma_lane_1(out20, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 0, out20);

              out21 = vld1q_f32(out_row0_2 + 1);
              out21 = neon_vfma_lane_2(out21, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 1, out21);

              out22 = vld1q_f32(out_row0_2 + 2);
              out22 = neon_vfma_lane_3(out22, in_vec, k02_vec);
              vst1q_f32(out_row0_2 + 2, out22);

              in += 4;
              out_row0_0 += 4;
              out_row0_1 += 4;
              out_row0_2 += 4;
            }
#endif
              for (; j < w; ++j) {
                float val = in[0];
                for (int k = 0; k < 3; ++k) {
                  out_row0_0[k] += val * k0_0[k];
                  out_row0_1[k] += val * k0_1[k];
                  out_row0_2[k] += val * k0_2[k + 1];
                }
                in++;
                out_row0_0++;
                out_row0_1++;
                out_row0_2++;
              }
            }
          }
        }
      }
    }
  }
}

void GroupDeconv2dNeonK3x3S2(const float *input,
                             const float *filter,
                             const int group,
                             const index_t *in_shape,
                             const index_t *out_shape,
                             float *output) {
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];

  const index_t in_img_size = h * w;
  const index_t out_img_size = outh * outw;

  const index_t inch_g = inch / group;
  const index_t outch_g = outch / group;

#pragma omp parallel for collapse(3) schedule(runtime)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (int g = 0; g < group; ++g) {
      for (index_t oc = 0; oc < outch_g; ++oc) {
        const index_t out_offset = b * outch + outch_g * g + oc;
        float *out_base = output + out_offset * out_img_size;
        for (index_t ic = 0; ic < inch_g; ++ic) {
          const index_t in_offset = b * inch + inch_g * g + ic;
          const float *input_base = input + in_offset * in_img_size;
          const index_t kernel_offset = (oc * group + g) * inch_g + ic;
          const float *kernel_base = filter + kernel_offset * 9;
          const float *in = input_base;

          const float *k0 = kernel_base;
          const float *k1 = kernel_base + 3;
          const float *k2 = kernel_base + 5;

#if defined(MACE_ENABLE_NEON)
          float32x4_t k0_vec = vld1q_f32(k0);
        float32x4_t k1_vec = vld1q_f32(k1);
        float32x4_t k2_vec = vld1q_f32(k2);
#endif
          for (index_t i = 0; i < h; ++i) {
            float *out_row_base = out_base + i * 2 * outw;
            float *out_row_0 = out_row_base;
            float *out_row_1 = out_row_0 + outw;
            float *out_row_2 = out_row_1 + outw;

            index_t j = 0;
#if defined(MACE_ENABLE_NEON)
            for (index_t n = 0; n + 9 < outw; n += 8) {
            float32x4_t in_vec = vld1q_f32(in);

            // out row 0
            float32x4x2_t out00 = vld2q_f32(out_row_0);
            out00.val[0] =
              neon_vfma_lane_0(out00.val[0], in_vec, k0_vec);
            out00.val[1] =
              neon_vfma_lane_1(out00.val[1], in_vec, k0_vec);
            vst2q_f32(out_row_0, out00);

            float32x4x2_t out01 = vld2q_f32(out_row_0 + 2);
            out01.val[0] =
              neon_vfma_lane_2(out01.val[0], in_vec, k0_vec);
            vst2q_f32(out_row_0 + 2, out01);

            // out row 1
            float32x4x2_t out10 = vld2q_f32(out_row_1);
            out10.val[0] =
              neon_vfma_lane_0(out10.val[0], in_vec, k1_vec);
            out10.val[1] =
              neon_vfma_lane_1(out10.val[1], in_vec, k1_vec);
            vst2q_f32(out_row_1, out10);

            float32x4x2_t out11 = vld2q_f32(out_row_1 + 2);
            out11.val[0] =
              neon_vfma_lane_2(out11.val[0], in_vec, k1_vec);
            vst2q_f32(out_row_1 + 2, out11);

            // out row 2
            float32x4x2_t out20 = vld2q_f32(out_row_2);
            out20.val[0] =
              neon_vfma_lane_1(out20.val[0], in_vec, k2_vec);
            out20.val[1] =
              neon_vfma_lane_2(out20.val[1], in_vec, k2_vec);
            vst2q_f32(out_row_2, out20);

            float32x4x2_t out21 = vld2q_f32(out_row_2 + 2);
            out21.val[0] =
              neon_vfma_lane_3(out21.val[0], in_vec, k2_vec);
            vst2q_f32(out_row_2 + 2, out21);

            in += 4;
            out_row_0 += 8;
            out_row_1 += 8;
            out_row_2 += 8;
            j += 4;
          }
#endif
            for (; j < w; ++j) {
              float val = in[0];

              for (int k = 0; k < 3; ++k) {
                out_row_0[k] += val * k0[k];
                out_row_1[k] += val * k1[k];
                out_row_2[k] += val * k2[k + 1];
              }

              in++;
              out_row_0 += 2;
              out_row_1 += 2;
              out_row_2 += 2;
            }
          }
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace mace
