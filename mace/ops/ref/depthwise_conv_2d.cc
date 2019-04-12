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


#include "mace/ops/ref/depthwise_conv_2d.h"

#include <vector>

namespace mace {
namespace ops {
namespace ref {

MaceStatus DepthwiseConv2d<float>::Compute(const OpContext *context,
                                           const Tensor *input,
                                           const Tensor *filter,
                                           Tensor *output) {
  MACE_UNUSED(context);

  const std::vector<index_t> in_shape = input->shape();
  const std::vector<index_t> filter_shape = filter->shape();
  std::vector<index_t> out_shape(4);

  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                 filter->shape().data(),
                                 dilations_.data(),
                                 strides_.data(),
                                 padding_type_,
                                 out_shape.data(),
                                 paddings.data());
  } else {
    paddings = paddings_;
    CalcNCHWOutputSize(input->shape().data(),
                       filter->shape().data(),
                       paddings_.data(),
                       dilations_.data(),
                       strides_.data(),
                       RoundType::FLOOR,
                       out_shape.data());
  }
  out_shape[1] *= in_shape[1];
  const index_t pad_top = paddings[0] >> 1;
  const index_t pad_left = paddings[1] >> 1;
  output->Resize(out_shape);

  const index_t multiplier = filter_shape[0];
  const index_t in_image_size = in_shape[2] * in_shape[3];
  const index_t out_image_size = out_shape[2] * out_shape[3];
  const index_t in_batch_size = in_shape[1] * in_image_size;
  const index_t out_batch_size = out_shape[1] * out_image_size;
  const index_t filter_size = filter_shape[2] * filter_shape[3];

  Tensor::MappingGuard input_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard output_guard(output);
  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto output_data = output->mutable_data<float>();

  for (index_t b = 0; b < in_shape[0]; b++) {
    for (index_t m = 0; m < out_shape[1]; ++m) {
      const index_t c = m / multiplier;
      const index_t multi_index = m % multiplier;

      const index_t in_height = in_shape[2];
      const index_t in_width = in_shape[3];
      const index_t out_height = out_shape[2];
      const index_t out_width = out_shape[3];
      const index_t in_channels = in_shape[1];

      float *out_ptr_base =
          output_data + b * out_batch_size + m * out_image_size;

      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          float sum = 0;

          const float *in_ptr_base =
              input_data + b * in_batch_size + c * in_image_size;
          const float *filter_ptr =
              filter_data + multi_index * in_channels * filter_size
                  + c * filter_size;

          for (index_t kh = 0; kh < filter_shape[2]; ++kh) {
            for (index_t kw = 0; kw < filter_shape[3]; ++kw) {
              const index_t
                  ih = -pad_top + h * strides_[0] + kh * dilations_[0];
              const index_t
                  iw = -pad_left + w * strides_[1] + kw * dilations_[1];
              if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                sum += in_ptr_base[ih * in_width + iw] * filter_ptr[kw];
              }
            }  // kw
            filter_ptr += filter_shape[3];
          }  // kh


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
