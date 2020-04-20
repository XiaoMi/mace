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

#include "mace/ops/arm/base/conv_2d.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace arm {

void Conv2dBase::CalOutputShapeAndInputPadSize(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    std::vector<index_t> *output_shape,
    std::vector<int> *in_pad_size) {
  if (paddings_.empty()) {
    CalcNCHWPaddingAndOutputSize(input_shape.data(),
                                 filter_shape.data(),
                                 dilations_.data(),
                                 strides_.data(),
                                 padding_type_,
                                 output_shape->data(),
                                 in_pad_size->data());
  } else {
    *in_pad_size = paddings_;
    CalcNCHWOutputSize(input_shape.data(),
                       filter_shape.data(),
                       paddings_.data(),
                       dilations_.data(),
                       strides_.data(),
                       RoundType::FLOOR,
                       output_shape->data());
  }
}

void Conv2dBase::CalOutputBoundaryWithoutUsingInputPad(
    const std::vector<index_t> &output_shape,
    const std::vector<int> in_pad_size,
    std::vector<index_t> *out_bound) {
  const int pad_top = in_pad_size[0] >> 1;
  const int pad_bottom = in_pad_size[0] - pad_top;
  const int pad_left = in_pad_size[1] >> 1;
  const int pad_right = in_pad_size[1] - pad_left;
  const index_t height = output_shape[2];
  const index_t width = output_shape[3];
  *out_bound = {
      pad_top == 0 ? 0 : (pad_top - 1) / strides_[0] + 1,
      pad_bottom == 0 ? height : height - ((pad_bottom - 1) / strides_[0] + 1),
      pad_left == 0 ? 0 : (pad_left - 1) / strides_[1] + 1,
      pad_right == 0 ? width : width - ((pad_right - 1) / strides_[1] + 1),
  };
}

void Conv2dBase::CalOutputShapeAndPadSize(const Tensor *input,
                                          const Tensor *filter,
                                          const int out_tile_height,
                                          const int out_tile_width,
                                          std::vector<index_t> *output_shape,
                                          std::vector<int> *in_pad_size,
                                          std::vector<int> *out_pad_size) {
  in_pad_size->resize(4);
  out_pad_size->resize(4);
  output_shape->resize(4);

  const index_t in_height = input->dim(2);
  const index_t in_width = input->dim(3);

  const index_t stride_h = strides_[0];
  const index_t stride_w = strides_[1];
  const index_t dilation_h = dilations_[0];
  const index_t dilation_w = dilations_[1];
  const index_t filter_h = filter->dim(2);
  const index_t filter_w = filter->dim(3);

  std::vector<int> paddings(2);
  CalOutputShapeAndInputPadSize(input->shape(),
                                filter->shape(),
                                output_shape,
                                &paddings);

  const index_t out_height = (*output_shape)[2];
  const index_t out_width = (*output_shape)[3];
  const index_t
      padded_out_height = RoundUp<index_t>(out_height, out_tile_height);
  const index_t padded_out_width = RoundUp<index_t>(out_width, out_tile_width);
  const index_t padded_in_height =
      std::max(in_height + paddings[0], (padded_out_height - 1) * stride_h
          + (filter_h - 1) * dilation_h + 1);
  const index_t padded_in_width =
      std::max(in_width + paddings[1], (padded_out_width - 1) * stride_w
          + (filter_w - 1) * dilation_w + 1);

  (*in_pad_size)[0] = paddings[0] >> 1;
  (*in_pad_size)[1] =
      static_cast<int>(padded_in_height - in_height - (*in_pad_size)[0]);
  (*in_pad_size)[2] = paddings[1] >> 1;
  (*in_pad_size)[3] =
      static_cast<int>(padded_in_width - in_width - (*in_pad_size)[2]);

  (*out_pad_size)[0] = 0;
  (*out_pad_size)[1] = static_cast<int>(padded_out_height - out_height);
  (*out_pad_size)[2] = 0;
  (*out_pad_size)[3] = static_cast<int>(padded_out_width - out_width);
}

MaceStatus Conv2dBase::ResizeOutAndPadInOut(const OpContext *context,
                                            const Tensor *input,
                                            const Tensor *filter,
                                            Tensor *output,
                                            const int out_tile_height,
                                            const int out_tile_width,
                                            std::unique_ptr<const Tensor>
                                            *padded_input,
                                            std::unique_ptr<Tensor>
                                            *padded_output) {
  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(input,
                           filter,
                           out_tile_height,
                           out_tile_width,
                           &output_shape,
                           &in_pad_size,
                           &out_pad_size);
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  const index_t batch = input->dim(0);
  const index_t in_channels = input->dim(1);
  const index_t in_height = input->dim(2);
  const index_t in_width = input->dim(3);
  const index_t out_channels = output->dim(1);
  const index_t out_height = output->dim(2);
  const index_t out_width = output->dim(3);

  const index_t padded_in_height = in_height + in_pad_size[0] + in_pad_size[1];
  const index_t padded_in_width = in_width + in_pad_size[2] + in_pad_size[3];
  const index_t
      padded_out_height = out_height + out_pad_size[0] + out_pad_size[1];
  const index_t
      padded_out_width = out_width + out_pad_size[2] + out_pad_size[3];
  const bool is_in_padded =
      padded_in_height != in_height || padded_in_width != in_width;
  const bool is_out_padded =
      padded_out_height != out_height || padded_out_width != out_width;

  auto scratch_buffer = context->device()->scratch_buffer();
  const index_t padded_in_size =
      MACE_EXTRA_BUFFER_PAD_SIZE + (is_in_padded ? PadAlignSize(
          type_size_ * batch * in_channels * padded_in_height
              * padded_in_width) : 0);
  const index_t padded_out_size = is_out_padded ? PadAlignSize(
      type_size_ * batch * out_channels * padded_out_height
          * padded_out_width) : 0;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(padded_in_size + padded_out_size);
  if (is_in_padded) {
    std::unique_ptr<Tensor>
        padded_in =
        make_unique<Tensor>(scratch_buffer->Scratch(padded_in_size),
                            input->dtype());
    padded_in->Resize({batch, in_channels, padded_in_height, padded_in_width});
    PadInput(*input, in_pad_size[0], in_pad_size[2], padded_in.get());
    *padded_input = std::move(padded_in);
  }
  if (is_out_padded) {
    std::unique_ptr<Tensor>
        padded_out =
        make_unique<Tensor>(scratch_buffer->Scratch(padded_out_size),
                            output->dtype());
    padded_out->Resize({batch, out_channels, padded_out_height,
                        padded_out_width});
    *padded_output = std::move(padded_out);
  }
  return MaceStatus::MACE_SUCCESS;
}

void Conv2dBase::PadInput(const Tensor &src,
                          const int pad_top,
                          const int pad_left,
                          Tensor *dst) {
  if (dst == &src) return;
  const index_t batch = src.dim(0);
  const index_t channels = src.dim(1);
  const index_t height = src.dim(2);
  const index_t width = src.dim(3);
  const index_t padded_height = dst->dim(2);
  const index_t padded_width = dst->dim(3);
  const int pad_bottom = static_cast<int>(padded_height - height - pad_top);
  const int pad_right = static_cast<int>(padded_width - width - pad_left);
  auto in_data = src.data<uint8_t>();
  auto padded_in_data = dst->mutable_data<uint8_t>();

  const index_t img_size = height * width;
  const index_t padded_img_size = padded_height * padded_width;

  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t bc = b * channels + c;
      const uint8_t *in_base = in_data + bc * img_size * type_size_;
      uint8_t *padded_in_base =
          padded_in_data + bc * padded_img_size * type_size_;

      memset(padded_in_base, 0, type_size_ * pad_top * padded_width);
      padded_in_base += pad_top * padded_width * type_size_;
      for (index_t h = 0; h < height; ++h) {
        memset(padded_in_base,
               0,
               type_size_ * pad_left);
        memcpy(padded_in_base + pad_left * type_size_,
               in_base,
               type_size_ * width);
        memset(padded_in_base + (pad_left + width) * type_size_,
               0,
               type_size_ * pad_right);
        in_base += width * type_size_;
        padded_in_base += padded_width * type_size_;
      }
      memset(padded_in_base, 0, type_size_ * pad_bottom * padded_width);
    }
  }
}

void Conv2dBase::UnPadOutput(const Tensor &src, Tensor *dst) {
  if (dst == &src) return;
  const index_t batch = dst->dim(0);
  const index_t channels = dst->dim(1);
  const index_t height = dst->dim(2);
  const index_t width = dst->dim(3);
  const index_t padded_height = src.dim(2);
  const index_t padded_width = src.dim(3);

  auto padded_out_data = src.data<uint8_t>();
  auto out_data = dst->mutable_data<uint8_t>();

  const index_t img_size = height * width;
  const index_t padded_img_size = padded_height * padded_width;

  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t bc = (b * channels + c);
      uint8_t *out_base = out_data + bc * img_size * type_size_;
      const uint8_t *padded_out_base =
          padded_out_data + bc * padded_img_size * type_size_;

      for (index_t h = 0; h < height; ++h) {
        memcpy(out_base, padded_out_base, type_size_ * width);
        out_base += width * type_size_;
        padded_out_base += padded_width * type_size_;
      }  // h
    }  // c
  }  // b
}

ConvComputeParam Conv2dBase::PreWorkAndGetConv2DParam(
    const OpContext *context, const Tensor *in_tensor, Tensor *out_tensor) {
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

  return ConvComputeParam(batch, in_channels, in_height, in_width,
                          out_channels, out_height, out_width,
                          in_image_size, out_image_size,
                          in_batch_size, out_batch_size, &thread_pool);
}

DepthwiseConvComputeParam Conv2dBase::PreWorkAndGetDepthwiseConv2DParam(
    const OpContext *context, const Tensor *input,
    const Tensor *filter, Tensor *output) {
  std::vector<index_t> out_shape(4);
  std::vector<int> paddings(2);
  auto &in_shape = input->shape();
  auto &filter_shape = filter->shape();
  CalOutputShapeAndInputPadSize(in_shape, filter_shape, &out_shape, &paddings);
  out_shape[1] *= filter_shape[1];
  MACE_CHECK(output->Resize(out_shape) == MaceStatus::MACE_SUCCESS,
             "Resize failed.");
  output->Clear();

  const int pad_top = paddings[0] / 2;
  const int pad_left = paddings[1] / 2;

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
  const index_t multiplier = out_channels / in_channels;

  std::vector<index_t> out_bounds;
  CalOutputBoundaryWithoutUsingInputPad(out_shape, paddings, &out_bounds);
  const index_t valid_h_start = out_bounds[0];
  const index_t valid_h_stop = out_bounds[1];
  const index_t valid_w_start = out_bounds[2];
  const index_t valid_w_stop = out_bounds[3];

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  return DepthwiseConvComputeParam(
      batch, in_channels, in_height, in_width, out_channels, out_height,
      out_width, in_image_size, out_image_size, in_batch_size, out_batch_size,
      &thread_pool, pad_top, pad_left, multiplier, valid_h_start, valid_h_stop,
      valid_w_start, valid_w_stop);
}

}  // namespace arm
}  // namespace ops
}  // namespace mace

