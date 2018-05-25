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

#ifndef MACE_KERNELS_TRANSPOSE_H_
#define MACE_KERNELS_TRANSPOSE_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static void TransposeNHWCToNCHWC3(const float *input,
                                  float *output,
                                  const index_t height,
                                  const index_t width) {
  index_t image_size = height * width;

#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width * 3;
    index_t out_offset = h * width;

#if defined(MACE_ENABLE_NEON)
    index_t w;
    for (w = 0; w + 3 < width; w += 4) {
      float32x4x3_t vi = vld3q_f32(input + in_offset);
      vst1q_f32(output + out_offset, vi.val[0]);
      vst1q_f32(output + out_offset + image_size, vi.val[1]);
      vst1q_f32(output + out_offset + image_size * 2, vi.val[2]);

      in_offset += 12;
      out_offset += 4;
    }
    for (; w < width; ++w) {
      for (index_t c = 0; c < 3; ++c) {
        output[h * width + image_size * c + w] =
          input[h * width * 3 + w * 3 + c];
      }
    }
#else
    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 3; ++c) {
        output[out_offset + c * image_size + w] = input[in_offset + w * 3 + c];
      }
    }
#endif
  }
}

static void TransposeNCHWToNHWCC2(const float *input,
                                  float *output,
                                  const index_t height,
                                  const index_t width) {
  index_t image_size = height * width;
#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width;
    index_t out_offset = h * width * 2;

#if defined(MACE_ENABLE_NEON)
    index_t w;
    for (w = 0; w + 3 < width; w += 4) {
      float32x4_t vi0 = vld1q_f32(input + in_offset);
      float32x4_t vi1 = vld1q_f32(input + in_offset + image_size);
      float32x4x2_t vi = {vi0, vi1};
      vst2q_f32(output + out_offset, vi);
      in_offset += 4;
      out_offset += 8;
    }
    for (; w < width; ++w) {
      for (index_t c = 0; c < 2; ++c) {
        output[h * width * 2 + w * 2 + c] =
          input[h * width + image_size * c + w];
      }
    }
#else
    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 2; ++c) {
        output[out_offset + w * 2 + c] = input[in_offset + c * image_size + w];
      }
    }
#endif
  }
}

template<DeviceType D, typename T>
struct TransposeFunctor {
  explicit TransposeFunctor(const std::vector<int> &dims) : dims_(dims) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const std::vector<index_t> &input_shape = input->shape();
    const std::vector<index_t> &output_shape = output->shape();
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    if (input->dim_size() == 2) {
      MACE_CHECK(dims_[0] == 1 && dims_[1] == 0, "no need transform");
      index_t stride_i = input_shape[0];
      index_t stride_j = input_shape[1];
      for (int i = 0; i < input_shape[0]; ++i) {
        for (int j = 0; j < input_shape[1]; ++j) {
          output_data[j * stride_i + i] = input_data[i * stride_j + j];
        }
      }
    } else if (input->dim_size() == 4) {
      std::vector<int> transpose_order_from_NHWC_to_NCHW{0, 3, 1, 2};
      std::vector<int> transpose_order_from_NCHW_to_NHWC{0, 2, 3, 1};
      index_t batch_size = input->dim(1) * input->dim(2) * input->dim(3);
      if (dims_ == transpose_order_from_NHWC_to_NCHW && input->dim(3) == 3) {
        for (index_t b = 0; b < input->dim(0); ++b) {
          TransposeNHWCToNCHWC3(input_data + b * batch_size,
                                output_data + b * batch_size,
                                input->dim(1),
                                input->dim(2));
        }
      } else if (dims_ == transpose_order_from_NCHW_to_NHWC
          && input->dim(1) == 2) {
        for (index_t b = 0; b < input->dim(0); ++b) {
          TransposeNCHWToNHWCC2(input_data + b * batch_size,
                                output_data + b * batch_size,
                                input->dim(2),
                                input->dim(3));
        }
      } else {
        std::vector<index_t>
            in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                      input_shape[2] * input_shape[3], input_shape[3], 1};
        std::vector<index_t>
            out_stride{output_shape[1] * output_shape[2] * output_shape[3],
                       output_shape[2] * output_shape[3], output_shape[3], 1};

        std::vector<index_t> idim(4, 0);
        std::vector<index_t> odim(4, 0);
        for (odim[0] = 0; odim[0] < output_shape[0]; ++odim[0]) {
          for (odim[1] = 0; odim[1] < output_shape[1]; ++odim[1]) {
            for (odim[2] = 0; odim[2] < output_shape[2]; ++odim[2]) {
              for (odim[3] = 0; odim[3] < output_shape[3]; ++odim[3]) {
                idim[dims_[0]] = odim[0];
                idim[dims_[1]] = odim[1];
                idim[dims_[2]] = odim[2];
                idim[dims_[3]] = odim[3];

                output_data[odim[0] * out_stride[0] + odim[1] * out_stride[1]
                    + odim[2] * out_stride[2] + odim[3]] =
                    input_data[idim[0] * in_stride[0] + idim[1] * in_stride[1]
                        + idim[2] * in_stride[2] + idim[3]];
              }
            }
          }
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MACE_SUCCESS;
  }

  std::vector<int> dims_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_TRANSPOSE_H_
