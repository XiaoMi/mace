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

#include "micro/ops/xtensa/conv_2d_xtensa.h"

#include "micro/base/logging.h"
#include "nnlib/xa_nnlib_api.h"

namespace micro {
namespace ops {

MaceStatus Conv2dXtensaOp::Compute(int32_t (&output_dims)[4]) {
  const int32_t batch = output_dims[0];
  MACE_ASSERT(batch == 1);
  const int32_t height = output_dims[1];
  const int32_t width = output_dims[2];
  const int32_t channel = output_dims[3];
  const int32_t k_height = filter_dims_[1];
  const int32_t k_width = filter_dims_[2];
  const int32_t k_channel = filter_dims_[3];
  MACE_ASSERT(filter_dims_[0] == channel && input_dims_[3] == k_channel);
  const int32_t in_height = input_dims_[1];
  const int32_t in_width = input_dims_[2];
  const int32_t in_channel = input_dims_[3];

  const int32_t pad_y = padding_sizes_[0] / 2;
  const int32_t pad_x = padding_sizes_[1] / 2;

  const int32_t stride_y = strides_[0];
  const int32_t stride_x = strides_[1];

  const int32_t scratch_size = xa_nn_conv2d_std_getsize(
      in_height, in_channel, k_height, k_width, stride_y, pad_y, height, -1);

  ScratchBuffer scratch_buffer(engine_config_);

  float *bias_data =
      const_cast<float *>(reinterpret_cast<const float *>(bias_));
  if (bias_data == NULL) {
    bias_data = scratch_buffer.GetBuffer<float>(channel);
    for (int32_t i = 0; i < channel; ++i) {
      bias_data[i] = 0;
    }
  }

  float *scratch = scratch_buffer.GetBuffer<float>(scratch_size);

  int32_t re = xa_nn_conv2d_std_f32(output_, input_, filter_, bias_data,
                                    in_height, in_width, in_channel, k_height,
                                    k_width, channel, stride_x, stride_y, pad_x,
                                    pad_y, height, width, 0, scratch);

  MACE_ASSERT(re == 0);

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
