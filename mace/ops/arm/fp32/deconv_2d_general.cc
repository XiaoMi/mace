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

#include "mace/ops/arm/fp32/deconv_2d_general.h"

// TODO(liutuo): optimize it

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Deconv2dGeneral::Compute(const OpContext *context,
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

  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_img_size = out_height * out_width;
  const index_t in_img_size = in_height * in_width;
  const index_t kernel_h = filter->dim(2);
  const index_t kernel_w = filter->dim(3);

  const int kernel_size = static_cast<int>(kernel_h * kernel_w);
  std::vector<index_t> index_map(kernel_size, 0);
  for (index_t i = 0; i < kernel_h; ++i) {
    for (index_t j = 0; j < kernel_w; ++j) {
      index_map[i * kernel_w + j] = i * out_width + j;
    }
  }

  const index_t batch = in_shape[0];
  const index_t out_channels = out_shape[1];
  const index_t in_channels = in_shape[1];

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t oc = start1; oc < end1; oc += step1) {
        float *out_base =
            padded_out_data + (b * out_channels + oc) * out_img_size;
        for (index_t i = 0; i < in_height; ++i) {
          for (index_t j = 0; j < in_width; ++j) {
            const index_t out_offset =
                i * strides_[0] * out_width + j * strides_[1];
            for (int ic = 0; ic < in_channels; ++ic) {
              const index_t input_idx =
                  (b * in_channels + ic) * in_img_size + i * in_width + j;
              const float val = input_data[input_idx];
              const index_t kernel_offset =
                  (oc * in_channels + ic) * kernel_size;
              for (int k = 0; k < kernel_size; ++k) {
                const index_t out_idx = out_offset + index_map[k];
                const index_t kernel_idx = kernel_offset + k;
                out_base[out_idx] += val * filter_data[kernel_idx];
              }
            }
          }
        }
      }
    }
  }, 0, batch, 1, 0, out_channels, 1);

  UnPadOutput(*out_tensor, out_pad_size, output);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

