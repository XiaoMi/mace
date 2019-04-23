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

#include <utility>
#include <memory>
#include <functional>
#include "mace/ops/ref/depthwise_deconv_2d.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace ref {

MaceStatus DepthwiseDeconv2d<float>::Compute(const OpContext *context,
                                             const Tensor *input,
                                             const Tensor *filter,
                                             const Tensor *output_shape,
                                             Tensor *output) {
  MACE_UNUSED(context);

  std::vector<index_t> out_shape;
  if (output_shape) {
    Tensor::MappingGuard out_shape_guard(output_shape);
    MACE_CHECK(output_shape->size() == 4, "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape->data<int32_t>(),
                             output_shape->data<int32_t>() + 4);
  }
  std::vector<index_t> padded_out_shape;
  std::vector<int> out_pad_size;
  CalDeconvOutputShapeAndPadSize(input->shape(),
                                 filter->shape(),
                                 strides_,
                                 padding_type_,
                                 paddings_,
                                 input->dim(1),
                                 &out_shape,
                                 nullptr,
                                 &out_pad_size,
                                 &padded_out_shape,
                                 framework_type_,
                                 DataFormat::NCHW);

  MACE_RETURN_IF_ERROR(output->Resize(out_shape));

  const bool is_out_padded =
      padded_out_shape[2] != out_shape[2]
          || padded_out_shape[3] != out_shape[3];

  std::unique_ptr<Tensor> padded_output(nullptr);
  if (is_out_padded) {
    index_t padded_out_size =
        std::accumulate(padded_out_shape.begin(),
                        padded_out_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    index_t scratch_size = PadAlignSize(padded_out_size);
    scratch->GrowSize(scratch_size);

    std::unique_ptr<Tensor>
        padded_out
        (make_unique<Tensor>(scratch->Scratch(scratch_size), DT_FLOAT));
    padded_out->Reshape(padded_out_shape);
    padded_output = std::move(padded_out);
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }

  out_tensor->Clear();

  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard output_mapper(output);

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto pad_out_data = out_tensor->mutable_data<float>();
  auto out_data = output->mutable_data<float>();

  auto &in_shape = input->shape();

  const index_t batch = in_shape[0];
  const index_t channels = in_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
  const index_t pad_out_height = padded_out_shape[2];
  const index_t pad_out_width = padded_out_shape[3];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_img_size = pad_out_height * pad_out_width;
  const index_t in_img_size = in_height * in_width;
  const index_t kernel_h = filter->dim(2);
  const index_t kernel_w = filter->dim(3);
  const int kernel_size = static_cast<int>(kernel_h * kernel_w);
  const index_t pad_top = out_pad_size[0] / 2;
  const index_t pad_left = out_pad_size[1] / 2;

  std::vector<int> index_map(kernel_size, 0);
  for (int i = 0; i < kernel_h; ++i) {
    for (int j = 0; j < kernel_w; ++j) {
      index_map[i * kernel_w + j] = i * pad_out_width + j;
    }
  }

  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      float *out_base =
          pad_out_data + (b * channels + c) * out_img_size;
      for (index_t i = 0; i < in_height; ++i) {
        for (index_t j = 0; j < in_width; ++j) {
          const index_t out_offset =
              i * strides_[0] * pad_out_width + j * strides_[1];
          const index_t input_idx =
              (b * channels + c) * in_img_size + i * in_width + j;
          const float val = input_data[input_idx];
          const index_t kernel_offset = c * kernel_size;
          for (int k = 0; k < kernel_size; ++k) {
            const index_t out_idx = out_offset + index_map[k];
            const index_t kernel_idx = kernel_offset + k;
            out_base[out_idx] += val * filter_data[kernel_idx];
          }
        }
      }
    }
  }

  if (out_tensor != output) {
    for (index_t i = 0; i < batch; ++i) {
      for (index_t j = 0; j < channels; ++j) {
        for (index_t k = 0; k < out_height; ++k) {
          const float *input_base =
              pad_out_data
                  + ((i * channels + j) * pad_out_height + (k + pad_top))
                      * pad_out_width;
          float *output_base =
              out_data + ((i * channels + j) * out_height + k) * out_width;
          memcpy(output_base, input_base + pad_left, out_width * sizeof(float));
        }
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus GroupDeconv2d<float>::Compute(const OpContext *context,
                                         const Tensor *input,
                                         const Tensor *filter,
                                         const Tensor *output_shape,
                                         Tensor *output) {
  MACE_UNUSED(context);

  std::vector<index_t> out_shape;
  if (output_shape) {
    Tensor::MappingGuard out_shape_guard(output_shape);
    MACE_CHECK(output_shape->size() == 4, "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape->data<int32_t>(),
                             output_shape->data<int32_t>() + 4);
  }
  std::vector<index_t> padded_out_shape;
  std::vector<int> out_pad_size;
  CalDeconvOutputShapeAndPadSize(input->shape(),
                                 filter->shape(),
                                 strides_,
                                 padding_type_,
                                 paddings_,
                                 group_,
                                 &out_shape,
                                 nullptr,
                                 &out_pad_size,
                                 &padded_out_shape,
                                 framework_type_,
                                 DataFormat::NCHW);

  MACE_RETURN_IF_ERROR(output->Resize(out_shape));

  const bool is_out_padded =
      padded_out_shape[2] != out_shape[2]
          || padded_out_shape[3] != out_shape[3];

  std::unique_ptr<Tensor> padded_output(nullptr);
  if (is_out_padded) {
    index_t padded_out_size =
        std::accumulate(padded_out_shape.begin(),
                        padded_out_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    index_t scratch_size = PadAlignSize(padded_out_size);
    scratch->GrowSize(scratch_size);

    std::unique_ptr<Tensor>
        padded_out
        (make_unique<Tensor>(scratch->Scratch(scratch_size), DT_FLOAT));
    padded_out->Reshape(padded_out_shape);
    padded_output = std::move(padded_out);
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }

  out_tensor->Clear();

  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard output_mapper(output);

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto pad_out_data = out_tensor->mutable_data<float>();
  auto out_data = output->mutable_data<float>();

  auto &in_shape = input->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
  const index_t pad_out_height = padded_out_shape[2];
  const index_t pad_out_width = padded_out_shape[3];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_img_size = pad_out_height * pad_out_width;
  const index_t in_img_size = in_height * in_width;
  const index_t kernel_h = filter->dim(2);
  const index_t kernel_w = filter->dim(3);
  const int kernel_size = static_cast<int>(kernel_h * kernel_w);
  const index_t pad_top = out_pad_size[0] / 2;
  const index_t pad_left = out_pad_size[1] / 2;

  std::vector<int> index_map(kernel_size, 0);
  for (int i = 0; i < kernel_h; ++i) {
    for (int j = 0; j < kernel_w; ++j) {
      index_map[i * kernel_w + j] = i * out_width + j;
    }
  }

  const int in_channels_g = in_channels / group_;
  const int out_channels_g = out_channels / group_;
  for (int b = 0; b < in_shape[0]; ++b) {
    for (int g = 0; g < group_; ++g) {
      for (int p = 0; p < out_channels_g; ++p) {
        const index_t out_base =
            ((b * group_ + g) * out_channels_g + p) * out_img_size;
        for (int i = 0; i < in_height; ++i) {
          for (int j = 0; j < in_width; ++j) {
            const index_t out_offset =
                i * strides_[0] * out_width + j * strides_[1];
            for (int q = 0; q < in_channels_g; ++q) {
              const index_t in_base =
                  ((b * group_ + g) * in_channels_g + q) * in_img_size;
              const index_t in_offset =
                  in_base + i * in_width + j;
              const float val = input_data[in_offset];
              const index_t k_offset =
                  ((p * group_ + g) * in_channels_g + q) * kernel_size;
              for (int k = 0; k < kernel_size; ++k) {
                const index_t out_idx = out_base + out_offset + index_map[k];
                const float w = filter_data[k_offset + k];
                pad_out_data[out_idx] += val * w;
              }
            }
          }
        }
      }
    }
  }

  if (out_tensor != output) {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < out_channels; ++j) {
        for (int k = 0; k < out_height; ++k) {
          const float *input_base =
              pad_out_data
                  + ((i * out_channels + j) * pad_out_height + (k + pad_top))
                      * pad_out_width;
          float *output_base =
              out_data + ((i * out_channels + j) * out_height + k) * out_width;
          memcpy(output_base, input_base + pad_left, out_width * sizeof(float));
        }
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
