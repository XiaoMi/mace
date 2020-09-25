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

#include <arm_neon.h>

#include "mace/ops/arm/base/common_neon.h"
#include "mace/ops/arm/base/deconv_2d_3x3.h"

namespace mace {
namespace ops {
namespace arm {

template<>
MaceStatus Deconv2dK3x3S1<float>::DoCompute(
    const DeconvComputeParam &p, const float *filter_data,
    const float *input_data, float *padded_out_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t oc = start1; oc < end1; oc += step1) {
        if (oc + 1 < p.out_channels) {
          float *out_base0 =
              padded_out_data + (b * p.out_channels + oc) * p.out_img_size;
          float *out_base1 = out_base0 + p.out_img_size;
          for (index_t ic = 0; ic < p.in_channels; ++ic) {
            const float *input_base = input_data +
                (b * p.in_channels + ic) * p.in_height * p.in_width;
            const float *kernel_base0 =
                filter_data + (oc * p.in_channels + ic) * 9;
            const float *kernel_base1 = kernel_base0 + p.in_channels * 9;
            const float *in = input_base;

            // output channel 0
            const float *k0_0 = kernel_base0;
            const float *k0_1 = kernel_base0 + 3;
            const float *k0_2 = kernel_base0 + 5;
            // output channel 1
            const float *k1_0 = kernel_base1;
            const float *k1_1 = kernel_base1 + 3;
            const float *k1_2 = kernel_base1 + 5;

            // load filter
            float32x4_t k00_vec, k01_vec, k02_vec;
            float32x4_t k10_vec, k11_vec, k12_vec;

            k00_vec = vld1q_f32(k0_0);
            k01_vec = vld1q_f32(k0_1);
            k02_vec = vld1q_f32(k0_2);

            k10_vec = vld1q_f32(k1_0);
            k11_vec = vld1q_f32(k1_1);
            k12_vec = vld1q_f32(k1_2);

            for (index_t i = 0; i < p.in_height; ++i) {
              float *out_row_base0 = out_base0 + i * p.out_width;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + p.out_width;
              float *out_row0_2 = out_row_base0 + 2 * p.out_width;

              float *out_row_base1 = out_base1 + i * p.out_width;
              float *out_row1_0 = out_row_base1;
              float *out_row1_1 = out_row_base1 + p.out_width;
              float *out_row1_2 = out_row_base1 + 2 * p.out_width;

              index_t j = 0;

              for (; j + 3 < p.in_width; j += 4) {
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

              for (; j < p.in_width; ++j) {
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
          float *out_base0 = padded_out_data +
              (b * p.out_channels + oc) * p.out_height * p.out_width;
          for (index_t ic = 0; ic < p.in_channels; ++ic) {
            const float *input_base = input_data +
                (b * p.in_channels + ic) * p.in_height * p.in_width;
            const float *kernel_base0 =
                filter_data + (oc * p.in_channels + ic) * 9;
            const float *in = input_base;
            const float *k0_0 = kernel_base0;
            const float *k0_1 = kernel_base0 + 3;
            const float *k0_2 = kernel_base0 + 5;

            // load filter
            float32x4_t k00_vec = vld1q_f32(k0_0);
            float32x4_t k01_vec = vld1q_f32(k0_1);
            float32x4_t k02_vec = vld1q_f32(k0_2);

            for (index_t i = 0; i < p.in_height; ++i) {
              float *out_row_base0 = out_base0 + i * p.out_width;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + p.out_width;
              float *out_row0_2 = out_row_base0 + 2 * p.out_width;
              index_t j = 0;

              for (; j + 3 < p.in_width; j += 4) {
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

              for (; j < p.in_width; ++j) {
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
  }, 0, p.batch, 1, 0, p.out_channels, 2);

  return MaceStatus::MACE_SUCCESS;
}

template<>
MaceStatus Deconv2dK3x3S2<float>::DoCompute(
    const DeconvComputeParam &p, const float *filter_data,
    const float *input_data, float *padded_out_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t oc = start1; oc < end1; oc += step1) {
        float *out_base =
            padded_out_data + (b * p.out_channels + oc) * p.out_img_size;
        for (index_t ic = 0; ic < p.in_channels; ++ic) {
          const float *input_base =
              input_data + (b * p.in_channels + ic) * p.in_height * p.in_width;
          const float *kernel_base =
              filter_data + (oc * p.in_channels + ic) * 9;
          const float *in = input_base;

          const float *k0 = kernel_base;
          const float *k1 = kernel_base + 3;
          const float *k2 = kernel_base + 5;

          float32x4_t k0_vec = vld1q_f32(k0);
          float32x4_t k1_vec = vld1q_f32(k1);
          float32x4_t k2_vec = vld1q_f32(k2);

          for (index_t i = 0; i < p.in_height; ++i) {
            float *out_row_base = out_base + i * 2 * p.out_width;
            float *out_row_0 = out_row_base;
            float *out_row_1 = out_row_0 + p.out_width;
            float *out_row_2 = out_row_1 + p.out_width;

            index_t j = 0;

            for (index_t n = 0; n + 9 < p.out_width; n += 8) {
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

            for (; j < p.in_width; ++j) {
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
  }, 0, p.batch, 1, 0, p.out_channels, 1);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
