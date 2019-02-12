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


#include "mace/ops/ref/conv_2d.h"

#include <vector>
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace ref {

MaceStatus Conv2d<float>::Compute(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output) {
  MACE_UNUSED(context);

  const std::vector<index_t> in_shape = input->shape();
  const std::vector<index_t> filter_shape = filter->shape();
  const std::vector<index_t> out_shape = output->shape();
  const std::vector<int> stride_hw{stride_h_, stride_w_};
  const std::vector<int> dilation_hw{dilation_h_, dilation_w_};
  const std::vector<int> paddings{pad_h_, pad_w_};
  const index_t pad_top = pad_h_ >> 1;
  const index_t pad_left = pad_w_ >> 1;

  std::vector<index_t> output_shape(4);

  CalcOutputSize(in_shape.data(),
                 NCHW,
                 filter_shape.data(),
                 OIHW,
                 paddings.data(),
                 dilation_hw.data(),
                 stride_hw.data(),
                 RoundType::FLOOR,
                 output_shape.data());
  output->Resize(output_shape);

  const index_t in_image_size = in_shape[2] * in_shape[3];
  const index_t out_image_size = out_shape[2] * out_shape[3];
  const index_t in_batch_size = filter_shape[1] * in_image_size;
  const index_t out_batch_size = filter_shape[0] * out_image_size;
  const index_t filter_size = filter_shape[2] * filter_shape[3];
  Tensor::MappingGuard input_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard output_guard(output);
  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto output_data = output->mutable_data<float>();

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < in_shape[0]; b++) {
    for (index_t m = 0; m < filter_shape[0]; ++m) {
      const index_t in_height = in_shape[2];
      const index_t in_width = in_shape[3];
      const index_t out_height = out_shape[2];
      const index_t out_width = out_shape[3];
      const index_t in_channels = filter_shape[1];

      float *out_ptr_base =
          output_data + b * out_batch_size + m * out_image_size;

      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          float sum = 0;

          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float *filter_ptr =
                filter_data + m * in_channels * filter_size + c * filter_size;

            for (index_t kh = 0; kh < filter_shape[2]; ++kh) {
              for (index_t kw = 0; kw < filter_shape[3]; ++kw) {
                const index_t ih = -pad_top + h * stride_h_ + kh * dilation_h_;
                const index_t iw = -pad_left + w * stride_w_ + kw * dilation_w_;
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                  sum += in_ptr_base[ih * in_width + iw] * filter_ptr[kw];
                }
              }  // kw
              filter_ptr += filter_shape[3];
            }  // kh
          }  // c

          out_ptr_base[h * out_width + w] = sum;
        }  // w
      }  // h
    }  // m
  }  // b
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace ref
}  // namespace ops
}  // namespace mace


