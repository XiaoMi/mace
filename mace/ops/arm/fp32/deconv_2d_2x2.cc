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

#include "mace/ops/arm/fp32/deconv_2d_2x2.h"

#include <arm_neon.h>
#include "mace/ops/arm/fp32/common_neon.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Deconv2dK2x2S1::Compute(const OpContext *context,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   const Tensor *output_shape,
                                   Tensor *output) {
  std::unique_ptr<Tensor> padded_out;
  std::vector<int> out_pad_size;
  ResizeOutAndPadOut(context,
                     input,
                     filter,
                     output_shape,
                     output,
                     &out_pad_size,
                     &padded_out);

  Tensor *out_tensor = output;
  if (padded_out != nullptr) {
    out_tensor = padded_out.get();
  }

  out_tensor->Clear();

  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard output_mapper(output);

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto padded_out_data = out_tensor->mutable_data<float>();

  auto &in_shape = input->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];

  const index_t out_img_size = outh * outw;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t oc = start1; oc < end1; oc += step1) {
        if (oc + 1 < outch) {
          float *out_base0 = padded_out_data + (b * outch + oc) * out_img_size;
          float *out_base1 = out_base0 + out_img_size;
          for (index_t ic = 0; ic < inch; ++ic) {
            const float *input_base = input_data + (b * inch + ic) * h * w;
            const float *kernel_base0 = filter_data + (oc * inch + ic) * 4;
            const float *kernel_base1 = kernel_base0 + inch * 4;
            const float *in = input_base;
            // output channel 0
            const float *k0 = kernel_base0;
            // output channel 1
            const float *k1 = kernel_base1;
            // load filter
            float32x4_t k0_vec = vld1q_f32(k0);
            float32x4_t k1_vec = vld1q_f32(k1);

            for (index_t i = 0; i < h; ++i) {
              float *out_row_base0 = out_base0 + i * outw;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + outw;

              float *out_row_base1 = out_base1 + i * outw;
              float *out_row1_0 = out_row_base1;
              float *out_row1_1 = out_row_base1 + outw;

              index_t j = 0;

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
          float *out_base0 = padded_out_data + (b * outch + oc) * outh * outw;
          for (index_t ic = 0; ic < inch; ++ic) {
            const float *input_base = input_data + (b * inch + ic) * h * w;
            const float *kernel_base0 = filter_data + (oc * inch + ic) * 4;
            const float *in = input_base;
            const float *k0 = kernel_base0;

            // load filter
            float32x4_t k0_vec = vld1q_f32(k0);

            for (index_t i = 0; i < h; ++i) {
              float *out_row_base0 = out_base0 + i * outw;
              float *out_row0_0 = out_row_base0;
              float *out_row0_1 = out_row_base0 + outw;
              index_t j = 0;

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
  }, 0, batch, 1, 0, outch, 2);

  UnPadOutput(*out_tensor, out_pad_size, output);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Deconv2dK2x2S2::Compute(const OpContext *context,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   const Tensor *output_shape,
                                   Tensor *output) {
  std::unique_ptr<Tensor> padded_out;
  std::vector<int> out_pad_size;
  ResizeOutAndPadOut(context,
                     input,
                     filter,
                     output_shape,
                     output,
                     &out_pad_size,
                     &padded_out);

  Tensor *out_tensor = output;
  if (padded_out != nullptr) {
    out_tensor = padded_out.get();
  }

  out_tensor->Clear();

  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard output_mapper(output);

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto padded_out_data = out_tensor->mutable_data<float>();

  auto &in_shape = input->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];
  const index_t out_img_size = outh * outw;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t oc = start1; oc < end1; oc += step1) {
        float *out_base = padded_out_data + (b * outch + oc) * out_img_size;
        for (index_t ic = 0; ic < inch; ++ic) {
          const float *input_base = input_data + (b * inch + ic) * h * w;
          const float *kernel_base = filter_data + (oc * inch + ic) * 4;
          const float *in = input_base;
          const float *k0 = kernel_base;
          float32x4_t k0_vec = vld1q_f32(k0);

          for (index_t i = 0; i < h; ++i) {
            float *out_row_base = out_base + i * 2 * outw;
            float *out_row_0 = out_row_base;
            float *out_row_1 = out_row_0 + outw;

            index_t j = 0;

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
  }, 0, batch, 1, 0, outch, 1);


  UnPadOutput(*out_tensor, out_pad_size, output);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
