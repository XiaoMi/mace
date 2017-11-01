//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

void CalcPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                              const index_t *filter_shape,  // OIHW
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
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  padding_size[0] = 0;
  padding_size[1] = 0;

  index_t output_height = 0, output_width = 0;
  index_t kernel_height = filter_shape[2];
  index_t kernel_width = filter_shape[3];
  index_t output_channels = filter_shape[0];

  index_t k_extent_height = (kernel_height - 1) * dilations[0] + 1;
  index_t k_extent_width = (kernel_width - 1) * dilations[1] + 1;

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
  // TODO may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.

  padding_size[0] =
      (output_height - 1) * strides[0] + k_extent_height - input_shape[2];
  padding_size[1] =
      (output_width - 1) * strides[1] + k_extent_width - input_shape[3];

  output_shape[0] = input_shape[0];
  output_shape[1] = output_channels;
  output_shape[2] = output_height;
  output_shape[3] = output_width;
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
  // TODO may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.
  padding_size[0] =
      (output_height - 1) * strides[0] + k_extent_height - input_shape[2];
  padding_size[1] =
      (output_width - 1) * strides[1] + k_extent_width - input_shape[3];
}

void ConstructInputWithPadding(const float *input,
                               const index_t *input_shape,
                               const int *paddings,
                               Tensor *output_tensor) {
  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  std::vector<index_t> output_shape(
      {batch, channels, paddings[0] + height, paddings[1] + width});

  const index_t output_width = output_shape[3];
  const int padded_top = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  output_tensor->Resize(output_shape);

  Tensor::MappingGuard padded_input_mapper(output_tensor);
  float *output_ptr = output_tensor->mutable_data<float>();
  memset(output_ptr, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  output_ptr += padded_top * output_width;
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        memcpy(output_ptr + padded_left, input, width * sizeof(float));
        input += width;
        output_ptr += output_width;
      }
      // Skip the padded bottom in this channel and top in the next channel
      output_ptr += paddings[0] * output_width;
    }
  }
}
}  //  namespace kernels
}  //  namespace mace
