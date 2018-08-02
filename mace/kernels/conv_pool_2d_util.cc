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

#include "mace/kernels/conv_pool_2d_util.h"

#include <algorithm>
#include <vector>

namespace mace {
namespace kernels {

void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
      (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(output_shape);
  MACE_CHECK_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  padding_size[0] = 0;
  padding_size[1] = 0;
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];
  index_t k_extent_height = (kernel_height - 1) * dilations[0] + 1;
  index_t k_extent_width = (kernel_width - 1) * dilations[1] + 1;

  switch (padding) {
    case VALID:
      output_height = (input_height - k_extent_height) / strides[0] + 1;
      output_width = (input_width - k_extent_width) / strides[1] + 1;
      break;
    case SAME:
      output_height = (input_height - 1) / strides[0] + 1;
      output_width = (input_width - 1) / strides[1] + 1;
      break;
    case FULL:
      output_height = (input_height + k_extent_height - 2) / strides[0] + 1;
      output_width = (input_width + k_extent_width - 2) / strides[1] + 1;
      break;
    default:
      MACE_CHECK(false, "Unsupported padding type: ", padding);
  }

  // Note: TensorFlow may padded one more on the right/bottom side
  // TODO(liuqi): may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.

  padding_size[0] = std::max<int>(
      0, (output_height - 1) * strides[0] + k_extent_height - input_height);
  padding_size[1] = std::max<int>(
      0, (output_width - 1) * strides[1] + k_extent_width - input_width);

  output_shape[0] = input_shape[0];
  if (input_format == NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape, NCHW, filter_shape, OIHW, dilations,
                           strides, padding, output_shape, padding_size);
}

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,   // NHWC
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape, NHWC, filter_shape, OIHW, dilations,
                           strides, padding, output_shape, padding_size);
}

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
      (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(output_shape);
  MACE_CHECK_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];

  if (round_type == FLOOR) {
    output_height = static_cast<index_t>(
        std::floor(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::floor(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
  } else {
    output_height = static_cast<index_t>(
        std::ceil(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::ceil(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
  }

  output_shape[0] = input_shape[0];
  if (input_format == NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  CalcOutputSize(input_shape, NHWC, filter_shape, OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalcNCHWOutputSize(const index_t *input_shape,   // NCHW
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  CalcOutputSize(input_shape, NCHW, filter_shape, OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalPaddingSize(const index_t *input_shape,   // NCHW
                    const index_t *filter_shape,  // OIHW
                    const int *dilations,
                    const int *strides,
                    Padding padding,
                    int *padding_size) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
                 (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(padding_size);

  index_t output_height = 0, output_width = 0;
  index_t k_extent_height = (filter_shape[2] - 1) * dilations[0] + 1;
  index_t k_extent_width = (filter_shape[3] - 1) * dilations[1] + 1;

  switch (padding) {
    case VALID:
      output_height = (input_shape[2] - k_extent_height) / strides[0] + 1;
      output_width = (input_shape[3] - k_extent_width) / strides[1] + 1;
      break;
    case SAME:
      output_height = (input_shape[2] - 1) / strides[0] + 1;
      output_width = (input_shape[3] - 1) / strides[1] + 1;
      break;
    case FULL:
      output_height = (input_shape[2] + k_extent_height - 2) / strides[0] + 1;
      output_width = (input_shape[3] + k_extent_width - 2) / strides[1] + 1;
      break;
    default:
      MACE_CHECK(false, "Unsupported padding type: ", padding);
  }

  // Note: TensorFlow may padded one more on the right/bottom side
  // TODO(liuqi): may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.
  padding_size[0] = std::max<int>(
      0, (output_height - 1) * strides[0] + k_extent_height - input_shape[2]);
  padding_size[1] = std::max<int>(
      0, (output_width - 1) * strides[1] + k_extent_width - input_shape[3]);
}


MaceStatus ConstructNCHWInputWithPadding(const Tensor *input_tensor,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value) {
  Tensor::MappingGuard input_mapper(input_tensor);
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  std::vector<index_t> output_shape(
    {batch, channels, paddings[0] + height, paddings[1] + width});

  const index_t output_width = output_shape[3];
  const int padded_top = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();
  memset(output_data, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  if (padding_same_value) {
#define MACE_COPY_INPUT                                                 \
  std::fill(output_data, output_data + padded_left, input[0]);          \
  output_data += padded_left;                                           \
  memcpy(output_data, input, width * sizeof(float));                    \
  output_data += width;                                                 \
  std::fill(output_data, output_data + padded_right, input[width - 1]); \
  output_data += padded_right;

    const int padded_bottom = paddings[0] - padded_top;
    const int padded_right = paddings[1] - padded_left;

    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        for (int k = 0; k < padded_top; ++k) {
          MACE_COPY_INPUT;
        }
        for (int k = 0; k < height; ++k) {
          MACE_COPY_INPUT;
          input += width;
        }
        input -= width;
        for (int k = 0; k < padded_bottom; ++k) {
          MACE_COPY_INPUT;
        }
        input += width;
      }
    }
#undef MACE_COPY_INPUT
  } else {
    output_data += padded_top * output_width;
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        for (int k = 0; k < height; ++k) {
          memcpy(output_data + padded_left, input, width * sizeof(float));
          input += width;
          output_data += output_width;
        }
        // Skip the padded bottom in this channel and top in the next channel
        output_data += paddings[0] * output_width;
      }
    }
  }

  return MACE_SUCCESS;
}

MaceStatus ConstructNCHWInputWithSpecificPadding(const Tensor *input_tensor,
                                           const int pad_top,
                                           const int pad_bottom,
                                           const int pad_left,
                                           const int pad_right,
                                           Tensor *output_tensor) {
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  const int pad_height = pad_top + pad_bottom;
  const int pad_width = pad_left + pad_right;
  std::vector<index_t> output_shape(
    {batch, channels, height + pad_height, width + pad_width});
  MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));
  output_tensor->Clear();
  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();

  const index_t output_height = output_shape[2];
  const index_t output_width = output_shape[3];
  const index_t in_image_size = height * width;
  const index_t out_image_size = output_height * output_width;
  const index_t in_batch_size = channels * in_image_size;
  const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        memcpy(output_data + i * out_batch_size + j * out_image_size
                 + (pad_top + k) * output_width + pad_left,
               input + i * in_batch_size + j * in_image_size + k * width,
               width * sizeof(float));
      }
      // Skip the padded bottom in this channel and top in the next channel
    }
  }

  return MACE_SUCCESS;
}


MaceStatus ConstructNHWCInputWithPadding(const Tensor *input_tensor,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value) {
  Tensor::MappingGuard input_mapper(input_tensor);
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t height = input_shape[1];
  index_t width = input_shape[2];
  index_t channels = input_shape[3];

  std::vector<index_t> output_shape(
      {batch, paddings[0] + height, paddings[1] + width, channels});

  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int padded_top = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();
  memset(output_data, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  if (padding_same_value) {
    LOG(FATAL) << "Not implemented";
  } else {
#pragma omp parallel for collapse(3)
    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          const float *input_ptr =
              input + ((n * height + h) * width + w) * channels;
          float *output_ptr =
              output_data +
              ((n * output_height + h + padded_top) * output_width + w +
               padded_left) *
                  channels;
          memcpy(output_ptr, input_ptr, channels * sizeof(float));
        }
      }
    }
  }

  return MACE_SUCCESS;
}

}  // namespace kernels
}  // namespace mace
