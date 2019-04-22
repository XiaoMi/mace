// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/common/conv_pool_2d_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mace {
namespace ops {

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
  if (input_format == DataFormat::NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == DataFormat::NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == DataFormat::OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == DataFormat::OHWI) {
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
    case SAME:output_height = (input_height - 1) / strides[0] + 1;
      output_width = (input_width - 1) / strides[1] + 1;
      break;
    case FULL:
      output_height = (input_height + k_extent_height - 2) / strides[0] + 1;
      output_width = (input_width + k_extent_width - 2) / strides[1] + 1;
      break;
    default:MACE_CHECK(false, "Unsupported padding type: ", padding);
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
  if (input_format == DataFormat::NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == DataFormat::NHWC) {
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
  CalcPaddingAndOutputSize(input_shape, DataFormat::NCHW, filter_shape,
                           DataFormat::OIHW, dilations,
                           strides, padding, output_shape, padding_size);
}

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,   // NHWC
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape, DataFormat::NHWC, filter_shape,
                           DataFormat::OIHW, dilations,
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
  if (input_format == DataFormat::NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == DataFormat::NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == DataFormat::OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == DataFormat::OHWI) {
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
  if (input_format == DataFormat::NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == DataFormat::NHWC) {
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
  CalcOutputSize(input_shape, DataFormat::NHWC, filter_shape,
                 DataFormat::OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalcNCHWOutputSize(const index_t *input_shape,   // NCHW
                        const index_t *filter_shape,  // OIHW
                        const int *padding_size,
                        const int *dilations,
                        const int *strides,
                        const RoundType round_type,
                        index_t *output_shape) {
  CalcOutputSize(input_shape, DataFormat::NCHW, filter_shape,
                 DataFormat::OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalcDeconvShape_TF(const std::vector<index_t> &input_shape,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<index_t> &output_shape,
                        const std::vector<int> &strides,
                        Padding padding_type,
                        const int group,
                        std::vector<int> *in_pad_size,
                        std::vector<int> *out_pad_size,
                        std::vector<index_t> *padded_out_shape,
                        DataFormat data_format) {
  const index_t
      in_height =
      data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t
      in_width =
          data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t
      out_height =
          data_format == DataFormat::NCHW ? output_shape[2] : output_shape[1];
  const index_t
      out_width =
          data_format == DataFormat::NCHW ? output_shape[3] : output_shape[2];

  const index_t extended_in_height = (in_height - 1) * strides[0] + 1;
  const index_t extended_in_width = (in_width - 1) * strides[1] + 1;

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t expected_input_height = 0, expected_input_width = 0;

  switch (padding_type) {
    case VALID:
      expected_input_height =
          (out_height - kernel_h + strides[0]) / strides[0];
      expected_input_width =
          (out_width - kernel_w + strides[1]) / strides[1];
      break;
    case SAME:
      expected_input_height =
          (out_height + strides[0] - 1) / strides[0];
      expected_input_width =
          (out_width + strides[1] - 1) / strides[1];
      break;
    default:MACE_CHECK(false, "Unsupported padding type: ", padding_type);
  }

  MACE_CHECK(expected_input_height == in_height,
             expected_input_height, "!=", in_height);
  MACE_CHECK(expected_input_width == in_width,
             expected_input_width, "!=", in_width);

  const index_t padded_out_height =
      (in_height - 1) * strides[0] + kernel_h;
  const index_t padded_out_width =
      (in_width - 1) * strides[1] + kernel_w;

  if (in_pad_size != nullptr) {
    const int p_h =
        static_cast<int>(out_height + kernel_h - 1 - extended_in_height);
    const int p_w =
        static_cast<int>(out_width + kernel_w - 1 - extended_in_width);
    in_pad_size->resize(2);
    (*in_pad_size)[0] = std::max<int>(0, p_h);
    (*in_pad_size)[1] = std::max<int>(0, p_w);
  }

  if (out_pad_size != nullptr) {
    const int o_p_h = static_cast<int>(padded_out_height - out_height);
    const int o_p_w = static_cast<int>(padded_out_width - out_width);
    out_pad_size->resize(2);
    (*out_pad_size)[0] = std::max<int>(0, o_p_h);
    (*out_pad_size)[1] = std::max<int>(0, o_p_w);
  }

  if (padded_out_shape != nullptr) {
    index_t output_channel = filter_shape[0] * group;
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = output_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }
}

void CalcDeconvShape_Caffe(const std::vector<index_t> &input_shape,
                           const std::vector<index_t> &filter_shape,
                           const std::vector<int> &strides,
                           const std::vector<int> &out_pad_size,
                           const int group,
                           std::vector<index_t> *out_shape,
                           std::vector<int> *in_pad_size,
                           std::vector<index_t> *padded_out_shape,
                           DataFormat data_format) {
  const index_t
      in_height =
          data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t
      in_width =
          data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t output_channel = filter_shape[0] * group;

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t padded_out_height =
      (in_height - 1) * strides[0] + kernel_h;
  index_t padded_out_width =
      (in_width - 1) * strides[1] + kernel_w;

  if (in_pad_size != nullptr) {
    in_pad_size->resize(2);
    (*in_pad_size)[0] = static_cast<int>((kernel_h - 1) * 2 - out_pad_size[0]);
    (*in_pad_size)[1] = static_cast<int>((kernel_w - 1) * 2 - out_pad_size[1]);
    (*in_pad_size)[0] = std::max<int>(0, (*in_pad_size)[0]);
    (*in_pad_size)[1] = std::max<int>(0, (*in_pad_size)[1]);
  }

  if (padded_out_shape != nullptr) {
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = input_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }

  if (out_shape != nullptr) {
    index_t out_height = padded_out_height - out_pad_size[0];
    index_t out_width = padded_out_width - out_pad_size[1];
    out_shape->resize(4);
    (*out_shape)[0] = input_shape[0];
    (*out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : out_height;
    (*out_shape)[2] = data_format == DataFormat::NCHW ? out_height : out_width;
    (*out_shape)[3] =
        data_format == DataFormat::NCHW ? out_width : output_channel;
  }
}

void CalDeconvOutputShapeAndPadSize(const std::vector<index_t> &input_shape,
                                    const std::vector<index_t> &filter_shape,
                                    const std::vector<int> &strides,
                                    Padding padding_type,
                                    const std::vector<int> &paddings,
                                    int group,
                                    std::vector<index_t> *output_shape,
                                    std::vector<int> *in_pad_size,
                                    std::vector<int> *out_pad_size,
                                    std::vector<index_t> *padded_out_shape,
                                    FrameworkType framework_type,
                                    DataFormat data_format) {
  if (framework_type == FrameworkType::TENSORFLOW) {
    MACE_CHECK(output_shape->size() == 4,
               "deconv output shape shoud be 4-dims");
    std::vector<index_t> &out_shape = *output_shape;
    if (data_format == DataFormat::NCHW) {
      const index_t t = out_shape[1];
      out_shape[1] = out_shape[3];
      out_shape[3] = out_shape[2];
      out_shape[2] = t;
    }

    CalcDeconvShape_TF(
        input_shape,
        filter_shape,
        *output_shape,
        strides,
        padding_type,
        group,
        in_pad_size,
        out_pad_size,
        padded_out_shape,
        data_format);
  } else {  // caffe
    if (!paddings.empty()) *out_pad_size = paddings;
    CalcDeconvShape_Caffe(
        input_shape,
        filter_shape,
        strides,
        *out_pad_size,
        group,
        output_shape,
        in_pad_size,
        padded_out_shape,
        data_format);
  }
}

}  // namespace ops
}  // namespace mace
