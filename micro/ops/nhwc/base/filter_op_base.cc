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

#include "micro/ops/nhwc/base/filter_op_base.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/model/argument.h"

namespace micro {
namespace ops {

MaceStatus FilterOpBase::OnInitBase() {
  strides_ = GetRepeatArgByName<int32_t>("strides");
  MACE_ASSERT(strides_ != NULL);

  const int32_t *dilations = GetRepeatArgByName<int32_t>("dilations");
  if (dilations == NULL) {
    dilations_[0] = dilations_[1] = 1;
  } else {
    base::memcpy(dilations_, dilations, 2 * sizeof(int32_t));
  }

  const int32_t *padding_sizes = GetRepeatArgByName<int32_t>("padding_values");
  if (padding_sizes == NULL) {
    padding_type_ = static_cast<Padding>(GetArgByName(
        "padding", static_cast<int32_t>(SAME)));
  } else {
    padding_type_ = NONE;
    base::memcpy(padding_sizes_, padding_sizes, 2 * sizeof(int32_t));
  }

  return MACE_SUCCESS;
}

void FilterOpBase::InitPaddingAndOutputSize(const int32_t *input_dims,
                                            const int32_t *filter_dims,
                                            const RoundType round_type,
                                            int32_t *output_dims) {
  if (padding_type_ != NONE) {
    CalcPaddingAndOutputSize(input_dims, filter_dims, output_dims);
  } else {
    CalcOutputSizeWithPaddingSize(
        input_dims, filter_dims, round_type, output_dims);
  }
}

void FilterOpBase::CalcPaddingAndOutputSize(const int32_t *input_dims,
                                            const int32_t *filter_dims,
                                            int32_t *output_dims) {
  MACE_ASSERT1(dilations_[0] > 0 && dilations_[1] > 0,
               "Invalid dilations, must >= 1");
  MACE_ASSERT1((dilations_[0] == 1 || strides_[0] == 1) &&
      (dilations_[1] == 1 || strides_[1] == 1),
               "If dilations > 1, strides should be 1");
  MACE_ASSERT(output_dims != NULL);

  int32_t input_height = input_dims[1];
  int32_t input_width = input_dims[2];
  int32_t kernel_height = filter_dims[1];
  int32_t kernel_width = filter_dims[2];
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  int32_t output_height = 0, output_width = 0;
  int32_t output_channels = filter_dims[0];
  int32_t k_extent_height = (kernel_height - 1) * dilations_[0] + 1;
  int32_t k_extent_width = (kernel_width - 1) * dilations_[1] + 1;

  switch (padding_type_) {
    case VALID: {
      output_height = (input_height - k_extent_height) / strides_[0] + 1;
      output_width = (input_width - k_extent_width) / strides_[1] + 1;
      break;
    }
    case SAME: {
      output_height = (input_height - 1) / strides_[0] + 1;
      output_width = (input_width - 1) / strides_[1] + 1;
      break;
    }
    case FULL: {
      output_height = (input_height + k_extent_height - 2) / strides_[0] + 1;
      output_width = (input_width + k_extent_width - 2) / strides_[1] + 1;
      break;
    }
    default: {
      MACE_ASSERT2(false, "Unsupported padding type: ",
                   static_cast<int32_t>(padding_type_));
      break;
    }
  }

  padding_sizes_[0] = base::max<int32_t>(
      0, (output_height - 1) * strides_[0] + k_extent_height - input_height);
  padding_sizes_[1] = base::max<int32_t>(
      0, (output_width - 1) * strides_[1] + k_extent_width - input_width);

  output_dims[0] = input_dims[0];
  output_dims[1] = output_height;
  output_dims[2] = output_width;
  output_dims[3] = output_channels;
}

void FilterOpBase::CalcOutputSizeWithPaddingSize(const int32_t *input_dims,
                                                 const int32_t *filter_dims,
                                                 const RoundType round_type,
                                                 int32_t *output_dims) {
  MACE_ASSERT1(dilations_[0] > 0 && dilations_[1] > 0,
               "Invalid dilations, must >= 1");
  MACE_ASSERT1((dilations_[0] == 1 || strides_[0] == 1) &&
      (dilations_[1] == 1 || strides_[1] == 1),
               "If dilations > 1, strides should be 1");
  MACE_ASSERT(output_dims != NULL);

  int32_t input_height = input_dims[1];
  int32_t input_width = input_dims[2];
  int32_t kernel_height = filter_dims[1];
  int32_t kernel_width = filter_dims[2];

  int32_t output_channels = filter_dims[0];
  float output_h_f = input_height + padding_sizes_[0] + padding_sizes_[0]
      - (kernel_height - 1) * dilations_[0] - 1;
  float output_w_f = input_width + padding_sizes_[1] + padding_sizes_[1]
      - (kernel_width - 1) * dilations_[1] - 1;
  int32_t output_height = 1;
  int32_t output_width = 1;
  if (round_type == FLOOR) {
    output_height += static_cast<int32_t>(output_h_f / strides_[0]);
    output_width += static_cast<int32_t>(output_w_f / strides_[1]);
  } else {
    output_height += base::ceil(output_h_f / strides_[0]);
    output_width += base::ceil(output_w_f / strides_[1]);
  }

  output_dims[0] = input_dims[0];
  output_dims[1] = output_height;
  output_dims[2] = output_width;
  output_dims[3] = output_channels;
}

}  // namespace ops
}  // namespace micro
