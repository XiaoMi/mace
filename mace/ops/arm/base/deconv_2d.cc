// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/base/deconv_2d.h"

#include <functional>
#include <utility>

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace arm {

MaceStatus Deconv2dBase::ResizeOutAndPadOut(
    const OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *output_shape,
    Tensor *output,
    std::vector<int> *out_pad_size,
    std::unique_ptr<Tensor> *padded_output) {
  std::vector<index_t> out_shape;
  if (output_shape) {
    MACE_CHECK(output_shape->size() == 4, "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape->data<int32_t>(),
                             output_shape->data<int32_t>() + 4);
  }

  std::vector<index_t> padded_out_shape;

  CalDeconvOutputShapeAndPadSize(input->shape(),
                                 filter->shape(),
                                 strides_,
                                 padding_type_,
                                 paddings_,
                                 group_,
                                 &out_shape,
                                 nullptr,
                                 out_pad_size,
                                 &padded_out_shape,
                                 framework_type_,
                                 DataFormat::NCHW);

  MACE_RETURN_IF_ERROR(output->Resize(out_shape));

  const bool is_out_padded =
      padded_out_shape[2] != out_shape[2]
          || padded_out_shape[3] != out_shape[3];

  if (is_out_padded) {
    auto *runtime = context->runtime();
    *padded_output = make_unique<Tensor>(
        runtime, output->dtype(), output->memory_type(), padded_out_shape);
    runtime->AllocateBufferForTensor(padded_output->get(), RENT_SCRATCH);
  }

  return MaceStatus::MACE_SUCCESS;
}

void Deconv2dBase::UnPadOutput(const Tensor &src,
                               const std::vector<int> &out_pad_size,
                               Tensor *dst) {
  if (dst == &src) return;
  const index_t pad_h = out_pad_size[0] / 2;
  const index_t pad_w = out_pad_size[1] / 2;

  const index_t batch = dst->dim(0);
  const index_t channels = dst->dim(1);
  const index_t height = dst->dim(2);
  const index_t width = dst->dim(3);
  const index_t padded_height = src.dim(2);
  const index_t padded_width = src.dim(3);

  auto padded_out_data = src.data<uint8_t>();
  auto out_data = dst->mutable_data<uint8_t>();

  for (index_t i = 0; i < batch; ++i) {
    for (index_t j = 0; j < channels; ++j) {
      for (index_t k = 0; k < height; ++k) {
        const uint8_t *input_base =
            padded_out_data + ((i * channels + j) * padded_height
                + (k + pad_h)) * padded_width * type_size_;
        uint8_t *output_base =
            out_data + ((i * channels + j) * height + k) * width * type_size_;
        memcpy(output_base,
               input_base + pad_w * type_size_,
               width * type_size_);
      }
    }
  }
}

DeconvComputeParam Deconv2dBase::PreWorkAndGetDeconvParam(
    const OpContext *context, const Tensor *input, Tensor *out_tensor) {

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

  utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

  return DeconvComputeParam(batch, inch, h, w, outch, outh, outw,
                            out_img_size, &thread_pool);
}

DepthwiseDeconvComputeParam Deconv2dBase::PreWorkAndGetDepthwiseDeconvParam(
    const OpContext *context, const Tensor *input, Tensor *out_tensor) {
  auto &in_shape = input->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t channels = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];
  const index_t in_img_size = h * w;
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];
  const index_t out_img_size = outh * outw;

  utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

  return DepthwiseDeconvComputeParam(batch, channels, h, w, in_img_size,
                                     outh, outw, out_img_size, &thread_pool);
}

GroupDeconvComputeParam Deconv2dBase::PreWorkAndGetGroupDeconvParam(
    const OpContext *context, const Tensor *input, Tensor *out_tensor) {
  auto &in_shape = input->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t inch = in_shape[1];
  const index_t h = in_shape[2];
  const index_t w = in_shape[3];

  const index_t outch = out_shape[1];
  const index_t outh = out_shape[2];
  const index_t outw = out_shape[3];

  const index_t in_img_size = h * w;
  const index_t out_img_size = outh * outw;

  const index_t inch_g = inch / group_;
  const index_t outch_g = outch / group_;

  utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

  return GroupDeconvComputeParam(batch, inch, h, w, outch, outh, outw,
                                 in_img_size, out_img_size, inch_g,
                                 outch_g, &thread_pool);
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
