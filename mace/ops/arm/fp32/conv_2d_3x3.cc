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

#include "mace/ops/arm/fp32/conv_2d_3x3.h"

#include <arm_neon.h>
#include <memory>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Conv2dK3x3S1::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       2,
                       4,
                       &padded_input,
                       &padded_output);
  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  Tensor::MappingGuard in_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard out_guard(output);
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 1 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base =
              output_data + b * out_batch_size + (m + 1) * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float
                *in_ptr0 = input_data + b * in_batch_size + c * in_image_size;
            const float
                *filter_ptr0 = filter_data + m * in_channels * 9 + c * 9;

            float *out_ptr1 = out_ptr1_base;
            const float *in_ptr1 =
                input_data + b * in_batch_size + c * in_image_size
                    + 1 * in_width;
            const float *in_ptr2 =
                input_data + b * in_batch_size + c * in_image_size
                    + 2 * in_width;
            const float *in_ptr3 =
                input_data + b * in_batch_size + c * in_image_size
                    + 3 * in_width;
            const float
                *filter_ptr1 = filter_data + (m + 1) * in_channels * 9 + c * 9;

#if defined(__aarch64__)
            float *out_ptr0 = out_ptr0_base;

            // load filter (2 outch x 3 height x 3 width): vf_outch_height
            float32x4_t vf00, vf01, vf02;
            float32x4_t vf10, vf11, vf12;
            vf00 = vld1q_f32(filter_ptr0);
            vf01 = vld1q_f32(filter_ptr0 + 3);
            vf02 = vld1q_f32(filter_ptr0 + 6);

            vf10 = vld1q_f32(filter_ptr1);
            vf11 = vld1q_f32(filter_ptr1 + 3);
            vf12 = vld1q_f32(filter_ptr1 + 6);

            for (index_t h = 0; h + 1 < out_height; h += 2) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input (4 height x 3 slide): vi_height_slide
                float32x4_t vi00, vi01, vi02;  // reg count: 14
                float32x4_t vi10, vi11, vi12;
                float32x4_t vi20, vi21, vi22;
                float32x4_t vi30, vi31, vi32;
                float32x4_t vo20, vo30;  // tmp use

                // output (4 outch x 2 height x 4 width): vo_outch_height
                float32x4_t vo00, vo01;
                float32x4_t vo10, vo11;

                // load input
                vi00 = vld1q_f32(in_ptr0);
                vo00 = vld1q_f32(in_ptr0 + 4);  // reuse vo00: vi0n
                vi10 = vld1q_f32(in_ptr1);
                vo10 = vld1q_f32(in_ptr1 + 4);
                vi20 = vld1q_f32(in_ptr2);
                vo20 = vld1q_f32(in_ptr2 + 4);
                vi30 = vld1q_f32(in_ptr3);
                vo30 = vld1q_f32(in_ptr3 + 4);

                vi01 = vextq_f32(vi00, vo00, 1);
                vi02 = vextq_f32(vi00, vo00, 2);
                vi11 = vextq_f32(vi10, vo10, 1);
                vi12 = vextq_f32(vi10, vo10, 2);
                vi21 = vextq_f32(vi20, vo20, 1);
                vi22 = vextq_f32(vi20, vo20, 2);
                vi31 = vextq_f32(vi30, vo30, 1);
                vi32 = vextq_f32(vi30, vo30, 2);

                // load ouptut
                vo00 = vld1q_f32(out_ptr0);
                vo01 = vld1q_f32(out_ptr0 + out_width);
                vo10 = vld1q_f32(out_ptr1);
                vo11 = vld1q_f32(out_ptr1 + out_width);

                // outch 0, height 0
                vo00 = vfmaq_laneq_f32(vo00, vi00, vf00, 0);  // reg count: 18
                vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 0);
                vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 2);

                // outch 0, height 1
                vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 2);

                // outch 1, height 0
                vo10 = vfmaq_laneq_f32(vo10, vi00, vf10, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi01, vf10, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi02, vf10, 2);
                vo10 = vfmaq_laneq_f32(vo10, vi10, vf11, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi11, vf11, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi12, vf11, 2);
                vo10 = vfmaq_laneq_f32(vo10, vi20, vf12, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi21, vf12, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi22, vf12, 2);

                // outch 1, height 1
                vo11 = vfmaq_laneq_f32(vo11, vi10, vf10, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi11, vf10, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi12, vf10, 2);
                vo11 = vfmaq_laneq_f32(vo11, vi20, vf11, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi21, vf11, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi22, vf11, 2);
                vo11 = vfmaq_laneq_f32(vo11, vi30, vf12, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi31, vf12, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi32, vf12, 2);

                vst1q_f32(out_ptr0, vo00);
                vst1q_f32(out_ptr0 + out_width, vo01);
                vst1q_f32(out_ptr1, vo10);
                vst1q_f32(out_ptr1 + out_width, vo11);

                in_ptr0 += 4;
                in_ptr1 += 4;
                in_ptr2 += 4;
                in_ptr3 += 4;

                out_ptr0 += 4;
                out_ptr1 += 4;
              }  // w

              in_ptr0 += 2 + in_width;
              in_ptr1 += 2 + in_width;
              in_ptr2 += 2 + in_width;
              in_ptr3 += 2 + in_width;

              out_ptr0 += out_width;
              out_ptr1 += out_width;
            }                      // h
#else  // arm v7
            float *out_ptr0 = out_ptr0_base;

            // load filter (2 outch x 3 height x 3 width): vf_outch_height
            float32x2_t vf001, vf023, vf045, vf067, vf089;
            float32x2_t vf101, vf123, vf145, vf167, vf189;
            vf001 = vld1_f32(filter_ptr0);
            vf023 = vld1_f32(filter_ptr0 + 2);
            vf045 = vld1_f32(filter_ptr0 + 4);
            vf067 = vld1_f32(filter_ptr0 + 6);
            vf089 = vld1_f32(filter_ptr0 + 8);

            vf101 = vld1_f32(filter_ptr1);
            vf123 = vld1_f32(filter_ptr1 + 2);
            vf145 = vld1_f32(filter_ptr1 + 4);
            vf167 = vld1_f32(filter_ptr1 + 6);
            vf189 = vld1_f32(filter_ptr1 + 8);

            for (index_t h = 0; h + 1 < out_height; h += 2) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input (4 height x 3 slide): vi_height_slide
                float32x4_t vi00, vi01, vi02;  // reg count: 14
                float32x4_t vi10, vi11, vi12;
                float32x4_t vi20, vi21, vi22;
                float32x4_t vi30, vi31, vi32;
                float32x4_t vo20, vo30;  // tmp use

                // output (4 outch x 2 height x 4 width): vo_outch_height
                float32x4_t vo00, vo01;
                float32x4_t vo10, vo11;

                // load input
                vi00 = vld1q_f32(in_ptr0);
                vo00 = vld1q_f32(in_ptr0 + 4);  // reuse vo00: vi0n
                vi10 = vld1q_f32(in_ptr1);
                vo10 = vld1q_f32(in_ptr1 + 4);
                vi20 = vld1q_f32(in_ptr2);
                vo20 = vld1q_f32(in_ptr2 + 4);
                vi30 = vld1q_f32(in_ptr3);
                vo30 = vld1q_f32(in_ptr3 + 4);

                vi01 = vextq_f32(vi00, vo00, 1);
                vi02 = vextq_f32(vi00, vo00, 2);
                vi11 = vextq_f32(vi10, vo10, 1);
                vi12 = vextq_f32(vi10, vo10, 2);
                vi21 = vextq_f32(vi20, vo20, 1);
                vi22 = vextq_f32(vi20, vo20, 2);
                vi31 = vextq_f32(vi30, vo30, 1);
                vi32 = vextq_f32(vi30, vo30, 2);

                // load ouptut
                vo00 = vld1q_f32(out_ptr0);
                vo01 = vld1q_f32(out_ptr0 + out_width);
                vo10 = vld1q_f32(out_ptr1);
                vo11 = vld1q_f32(out_ptr1 + out_width);

                // outch 0, height 0
                vo00 = vmlaq_lane_f32(vo00, vi00, vf001, 0);
                vo00 = vmlaq_lane_f32(vo00, vi01, vf001, 1);
                vo00 = vmlaq_lane_f32(vo00, vi02, vf023, 0);
                vo00 = vmlaq_lane_f32(vo00, vi10, vf023, 1);
                vo00 = vmlaq_lane_f32(vo00, vi11, vf045, 0);
                vo00 = vmlaq_lane_f32(vo00, vi12, vf045, 1);
                vo00 = vmlaq_lane_f32(vo00, vi20, vf067, 0);
                vo00 = vmlaq_lane_f32(vo00, vi21, vf067, 1);
                vo00 = vmlaq_lane_f32(vo00, vi22, vf089, 0);

                // outch 0, height 1
                vo01 = vmlaq_lane_f32(vo01, vi10, vf001, 0);
                vo01 = vmlaq_lane_f32(vo01, vi11, vf001, 1);
                vo01 = vmlaq_lane_f32(vo01, vi12, vf023, 0);
                vo01 = vmlaq_lane_f32(vo01, vi20, vf023, 1);
                vo01 = vmlaq_lane_f32(vo01, vi21, vf045, 0);
                vo01 = vmlaq_lane_f32(vo01, vi22, vf045, 1);
                vo01 = vmlaq_lane_f32(vo01, vi30, vf067, 0);
                vo01 = vmlaq_lane_f32(vo01, vi31, vf067, 1);
                vo01 = vmlaq_lane_f32(vo01, vi32, vf089, 0);

                // outch 1, height 0
                vo10 = vmlaq_lane_f32(vo10, vi00, vf101, 0);
                vo10 = vmlaq_lane_f32(vo10, vi01, vf101, 1);
                vo10 = vmlaq_lane_f32(vo10, vi02, vf123, 0);
                vo10 = vmlaq_lane_f32(vo10, vi10, vf123, 1);
                vo10 = vmlaq_lane_f32(vo10, vi11, vf145, 0);
                vo10 = vmlaq_lane_f32(vo10, vi12, vf145, 1);
                vo10 = vmlaq_lane_f32(vo10, vi20, vf167, 0);
                vo10 = vmlaq_lane_f32(vo10, vi21, vf167, 1);
                vo10 = vmlaq_lane_f32(vo10, vi22, vf189, 0);

                // outch 1, height 1
                vo11 = vmlaq_lane_f32(vo11, vi10, vf101, 0);
                vo11 = vmlaq_lane_f32(vo11, vi11, vf101, 1);
                vo11 = vmlaq_lane_f32(vo11, vi12, vf123, 0);
                vo11 = vmlaq_lane_f32(vo11, vi20, vf123, 1);
                vo11 = vmlaq_lane_f32(vo11, vi21, vf145, 0);
                vo11 = vmlaq_lane_f32(vo11, vi22, vf145, 1);
                vo11 = vmlaq_lane_f32(vo11, vi30, vf167, 0);
                vo11 = vmlaq_lane_f32(vo11, vi31, vf167, 1);
                vo11 = vmlaq_lane_f32(vo11, vi32, vf189, 0);

                vst1q_f32(out_ptr0, vo00);
                vst1q_f32(out_ptr0 + out_width, vo01);
                vst1q_f32(out_ptr1, vo10);
                vst1q_f32(out_ptr1 + out_width, vo11);

                in_ptr0 += 4;
                in_ptr1 += 4;
                in_ptr2 += 4;
                in_ptr3 += 4;

                out_ptr0 += 4;
                out_ptr1 += 4;
              }  // w

              in_ptr0 += 2 + in_width;
              in_ptr1 += 2 + in_width;
              in_ptr2 += 2 + in_width;
              in_ptr3 += 2 + in_width;

              out_ptr0 += out_width;
              out_ptr1 += out_width;
            }  // h
#endif
          }  // c
        } else {
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr0 =
                  input_data + b * in_batch_size + c * in_image_size;
              const float *in_ptr1 =
                  input_data + b * in_batch_size + c * in_image_size
                      + 1 * in_width;
              const float *in_ptr2 =
                  input_data + b * in_batch_size + c * in_image_size
                      + 2 * in_width;
              const float *in_ptr3 =
                  input_data + b * in_batch_size + c * in_image_size
                      + 3 * in_width;
              const float
                  *filter_ptr0 = filter_data + mm * in_channels * 9 + c * 9;

#if defined(__aarch64__)
              float *out_ptr0 = out_ptr0_base;

              // load filter (1 outch x 3 height x 3 width): vf_outch_height
              float32x4_t vf00, vf01, vf02;
              vf00 = vld1q_f32(filter_ptr0);
              vf01 = vld1q_f32(filter_ptr0 + 3);
              vf02 = vld1q_f32(filter_ptr0 + 5);

              for (index_t h = 0; h + 1 < out_height; h += 2) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input (4 height x 3 slide): vi_height_slide
                  float32x4_t vi00, vi01, vi02, vi0n;
                  float32x4_t vi10, vi11, vi12, vi1n;
                  float32x4_t vi20, vi21, vi22, vi2n;
                  float32x4_t vi30, vi31, vi32, vi3n;

                  // output (1 outch x 2 height x 4 width): vo_outch_height
                  float32x4_t vo00, vo01;

                  // load input
                  vi00 = vld1q_f32(in_ptr0);
                  vi0n = vld1q_f32(in_ptr0 + 4);
                  vi10 = vld1q_f32(in_ptr1);
                  vi1n = vld1q_f32(in_ptr1 + 4);
                  vi20 = vld1q_f32(in_ptr2);
                  vi2n = vld1q_f32(in_ptr2 + 4);
                  vi30 = vld1q_f32(in_ptr3);
                  vi3n = vld1q_f32(in_ptr3 + 4);

                  vi01 = vextq_f32(vi00, vi0n, 1);
                  vi02 = vextq_f32(vi00, vi0n, 2);
                  vi11 = vextq_f32(vi10, vi1n, 1);
                  vi12 = vextq_f32(vi10, vi1n, 2);
                  vi21 = vextq_f32(vi20, vi2n, 1);
                  vi22 = vextq_f32(vi20, vi2n, 2);
                  vi31 = vextq_f32(vi30, vi3n, 1);
                  vi32 = vextq_f32(vi30, vi3n, 2);

                  // load ouptut
                  vo00 = vld1q_f32(out_ptr0);
                  vo01 = vld1q_f32(out_ptr0 + out_width);

                  // outch 0, height 0
                  vo00 = vfmaq_laneq_f32(vo00, vi00, vf00, 0);
                  vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                  vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 3);

                  // outch 0, height 1
                  vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                  vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                  vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 3);

                  vst1q_f32(out_ptr0, vo00);
                  vst1q_f32(out_ptr0 + out_width, vo01);

                  in_ptr0 += 4;
                  in_ptr1 += 4;
                  in_ptr2 += 4;
                  in_ptr3 += 4;

                  out_ptr0 += 4;
                }  // w

                in_ptr0 += 2 + in_width;
                in_ptr1 += 2 + in_width;
                in_ptr2 += 2 + in_width;
                in_ptr3 += 2 + in_width;

                out_ptr0 += out_width;
              }                    // h
#else  // arm v7
              float *out_ptr0 = out_ptr0_base;

              // load filter (1 outch x 3 height x 3 width): vf_outch_height
              float32x2_t vf01, vf23, vf45, vf67, vf78;
              vf01 = vld1_f32(filter_ptr0);
              vf23 = vld1_f32(filter_ptr0 + 2);
              vf45 = vld1_f32(filter_ptr0 + 4);
              vf67 = vld1_f32(filter_ptr0 + 6);
              vf78 = vld1_f32(filter_ptr0 + 7);

              for (index_t h = 0; h + 1 < out_height; h += 2) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input (4 height x 3 slide): vi_height_slide
                  float32x4_t vi00, vi01, vi02, vi0n;
                  float32x4_t vi10, vi11, vi12, vi1n;
                  float32x4_t vi20, vi21, vi22, vi2n;
                  float32x4_t vi30, vi31, vi32, vi3n;

                  // output (1 outch x 2 height x 4 width): vo_outch_height
                  float32x4_t vo00, vo01;

                  // load input
                  vi00 = vld1q_f32(in_ptr0);
                  vi0n = vld1q_f32(in_ptr0 + 4);
                  vi10 = vld1q_f32(in_ptr1);
                  vi1n = vld1q_f32(in_ptr1 + 4);
                  vi20 = vld1q_f32(in_ptr2);
                  vi2n = vld1q_f32(in_ptr2 + 4);
                  vi30 = vld1q_f32(in_ptr3);
                  vi3n = vld1q_f32(in_ptr3 + 4);

                  vi01 = vextq_f32(vi00, vi0n, 1);
                  vi02 = vextq_f32(vi00, vi0n, 2);
                  vi11 = vextq_f32(vi10, vi1n, 1);
                  vi12 = vextq_f32(vi10, vi1n, 2);
                  vi21 = vextq_f32(vi20, vi2n, 1);
                  vi22 = vextq_f32(vi20, vi2n, 2);
                  vi31 = vextq_f32(vi30, vi3n, 1);
                  vi32 = vextq_f32(vi30, vi3n, 2);

                  // load ouptut
                  vo00 = vld1q_f32(out_ptr0);
                  vo01 = vld1q_f32(out_ptr0 + out_width);

                  // outch 0, height 0
                  vo00 = vmlaq_lane_f32(vo00, vi00, vf01, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi01, vf01, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi02, vf23, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi10, vf23, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi11, vf45, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi12, vf45, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi20, vf67, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi21, vf67, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi22, vf78, 1);

                  // outch 0, height 1
                  vo01 = vmlaq_lane_f32(vo01, vi10, vf01, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi11, vf01, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi12, vf23, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi20, vf23, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi21, vf45, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi22, vf45, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi30, vf67, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi31, vf67, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi32, vf78, 1);

                  vst1q_f32(out_ptr0, vo00);
                  vst1q_f32(out_ptr0 + out_width, vo01);

                  in_ptr0 += 4;
                  in_ptr1 += 4;
                  in_ptr2 += 4;
                  in_ptr3 += 4;

                  out_ptr0 += 4;
                }  // w

                in_ptr0 += 2 + in_width;
                in_ptr1 += 2 + in_width;
                in_ptr2 += 2 + in_width;
                in_ptr3 += 2 + in_width;

                out_ptr0 += out_width;
              }  // h
#endif
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, batch, 1, 0, out_channels, 2);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dK3x3S2::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;

  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);
  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  Tensor::MappingGuard in_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard out_guard(output);
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        for (index_t c = 0; c < in_channels; ++c) {
          const float
              *in_base = input_data + b * in_batch_size + c * in_image_size;
          const float *filter_ptr = filter_data + m * in_channels * 9 + c * 9;
          float
              *out_base = output_data + b * out_batch_size + m * out_image_size;

#if defined(__aarch64__)
          // load filter (1 outch x 3 height x 3 width): vf_outch_height
          float32x4_t vf00, vf01, vf02;
          vf00 = vld1q_f32(filter_ptr);
          vf01 = vld1q_f32(filter_ptr + 3);
          vf02 = vld1q_f32(filter_ptr + 5);

          for (index_t h = 0; h < out_height; ++h) {
            for (index_t w = 0; w + 3 < out_width; w += 4) {
              float32x4x2_t vi0, vi1, vi2;
              float32x4_t vi0n, vi1n, vi2n;

              // input (3 height x 3 slide): vi_height_slide
              float32x4_t vi00, vi01, vi02;
              float32x4_t vi10, vi11, vi12;
              float32x4_t vi20, vi21, vi22;

              // output (1 outch x 1 height x 4 width): vo
              float32x4_t vo;

              // load input
              index_t in_h = h * 2;
              index_t in_w = w * 2;
              index_t in_offset = in_h * in_width + in_w;
              vi0 = vld2q_f32(in_base + in_offset);  // [0.2.4.6, 1.3.5.7]
              vi1 = vld2q_f32(in_base + in_offset + in_width);
              vi2 = vld2q_f32(in_base + in_offset + 2 * in_width);

              vi0n = vld1q_f32(in_base + in_offset + 8);  // [8.9.10.11]
              vi1n = vld1q_f32(in_base + in_offset + in_width + 8);
              vi2n = vld1q_f32(in_base + in_offset + 2 * in_width + 8);

              // load ouptut
              index_t out_offset = h * out_width + w;
              vo = vld1q_f32(out_base + out_offset);

              vi00 = vi0.val[0];                // [0.2.4.6]
              vi01 = vi0.val[1];                // [1.3.5.7]
              vi02 = vextq_f32(vi00, vi0n, 1);  // [2.4.6.8]
              vi10 = vi1.val[0];
              vi11 = vi1.val[1];
              vi12 = vextq_f32(vi10, vi1n, 1);
              vi20 = vi2.val[0];
              vi21 = vi2.val[1];
              vi22 = vextq_f32(vi20, vi2n, 1);

              // outch 0, height 0
              vo = vfmaq_laneq_f32(vo, vi00, vf00, 0);
              vo = vfmaq_laneq_f32(vo, vi01, vf00, 1);
              vo = vfmaq_laneq_f32(vo, vi02, vf00, 2);
              vo = vfmaq_laneq_f32(vo, vi10, vf01, 0);
              vo = vfmaq_laneq_f32(vo, vi11, vf01, 1);
              vo = vfmaq_laneq_f32(vo, vi12, vf01, 2);
              vo = vfmaq_laneq_f32(vo, vi20, vf02, 1);
              vo = vfmaq_laneq_f32(vo, vi21, vf02, 2);
              vo = vfmaq_laneq_f32(vo, vi22, vf02, 3);

              vst1q_f32(out_base + out_offset, vo);
            }                      // w
          }                        // h
#else  // arm v7
          // load filter (1 outch x 3 height x 3 width): vf_outch_height
          float32x2_t vf01, vf23, vf45, vf67, vf78;
          vf01 = vld1_f32(filter_ptr);
          vf23 = vld1_f32(filter_ptr + 2);
          vf45 = vld1_f32(filter_ptr + 4);
          vf67 = vld1_f32(filter_ptr + 6);
          vf78 = vld1_f32(filter_ptr + 7);

          for (index_t h = 0; h < out_height; ++h) {
            for (index_t w = 0; w + 3 < out_width; w += 4) {
              float32x4x2_t vi0, vi1, vi2;
              float32x4_t vi0n, vi1n, vi2n;

              // input (3 height x 3 slide): vi_height_slide
              float32x4_t vi00, vi01, vi02;
              float32x4_t vi10, vi11, vi12;
              float32x4_t vi20, vi21, vi22;

              // output (1 outch x 1 height x 4 width): vo
              float32x4_t vo;

              // load input
              index_t in_h = h * 2;
              index_t in_w = w * 2;
              index_t in_offset = in_h * in_width + in_w;
              vi0 = vld2q_f32(in_base + in_offset);  // [0.2.4.6, 1.3.5.7]
              vi1 = vld2q_f32(in_base + in_offset + in_width);
              vi2 = vld2q_f32(in_base + in_offset + 2 * in_width);

              vi0n = vld1q_f32(in_base + in_offset + 8);  // [8.9.10.11]
              vi1n = vld1q_f32(in_base + in_offset + in_width + 8);
              vi2n = vld1q_f32(in_base + in_offset + 2 * in_width + 8);

              // load ouptut
              index_t out_offset = h * out_width + w;
              vo = vld1q_f32(out_base + out_offset);

              vi00 = vi0.val[0];                // [0.2.4.6]
              vi01 = vi0.val[1];                // [1.3.5.7]
              vi02 = vextq_f32(vi00, vi0n, 1);  // [2.4.6.8]
              vi10 = vi1.val[0];
              vi11 = vi1.val[1];
              vi12 = vextq_f32(vi10, vi1n, 1);
              vi20 = vi2.val[0];
              vi21 = vi2.val[1];
              vi22 = vextq_f32(vi20, vi2n, 1);

              // outch 0, height 0
              vo = vmlaq_lane_f32(vo, vi00, vf01, 0);
              vo = vmlaq_lane_f32(vo, vi01, vf01, 1);
              vo = vmlaq_lane_f32(vo, vi02, vf23, 0);
              vo = vmlaq_lane_f32(vo, vi10, vf23, 1);
              vo = vmlaq_lane_f32(vo, vi11, vf45, 0);
              vo = vmlaq_lane_f32(vo, vi12, vf45, 1);
              vo = vmlaq_lane_f32(vo, vi20, vf67, 0);
              vo = vmlaq_lane_f32(vo, vi21, vf67, 1);
              vo = vmlaq_lane_f32(vo, vi22, vf78, 1);

              vst1q_f32(out_base + out_offset, vo);
            }  // w
          }    // h
#endif
        }  // c
      }    // m
    }      // b
  }, 0, batch, 1, 0, out_channels, 1);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
