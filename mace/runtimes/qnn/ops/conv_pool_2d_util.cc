// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/qnn/ops/conv_pool_2d_util.h"

#include "mace/utils/logging.h"

namespace mace {

void CalcPadding(const uint32_t *input_shape,   // NHWC
                 const uint32_t *filter_shape,  // HWIO
                 const uint32_t *output_shape,
                 const int *dilations,
                 const int *strides,
                 const std::vector<int> &padding_values,
                 std::vector<uint32_t> *qnn_paddings) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must > 0");
  MACE_CHECK(strides[0] > 0 && strides[1] > 0, "Invalid strides, must > 0");
  MACE_CHECK(padding_values.size() == 0 || padding_values.size() == 2);
  MACE_CHECK_NOTNULL(qnn_paddings);
  MACE_CHECK_NOTNULL(input_shape);

  int padding_height = 0, padding_width = 0;
  if (padding_values.empty()) {
    index_t input_height = input_shape[1];
    index_t input_width = input_shape[2];
    index_t kernel_height = filter_shape[0];
    index_t kernel_width = filter_shape[1];
    index_t output_height = output_shape[1];
    index_t output_width = output_shape[2];
    index_t k_extent_height = (kernel_height - 1) * dilations[0] + 1;
    index_t k_extent_width = (kernel_width - 1) * dilations[1] + 1;
    padding_height = std::max<int>(
        0, (output_height - 1) * strides[0] + k_extent_height - input_height);
    padding_width = std::max<int>(
        0, (output_width - 1) * strides[1] + k_extent_width - input_width);
  } else {
    padding_height = padding_values[0];
    padding_width = padding_values[1];
  }
  uint32_t pad_top = padding_height / 2;
  uint32_t pad_bottom = padding_height - pad_top;
  uint32_t pad_left = padding_width / 2;
  uint32_t pad_right = padding_width - pad_left;

  qnn_paddings->assign({pad_top, pad_bottom, pad_left, pad_right});
}
}  // namespace mace
