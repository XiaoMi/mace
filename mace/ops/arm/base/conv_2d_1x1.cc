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

#include "mace/ops/arm/base/conv_2d_1x1.h"

#include <vector>

namespace mace {
namespace ops {
namespace arm {

template<typename T>
MaceStatus Conv2dK1x1<T>::Compute(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output) {
  index_t batch = input->dim(0);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);
  index_t in_channels = input->dim(1);

  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(input, filter, 1, 1,
                           &output_shape, &in_pad_size, &out_pad_size);
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  const index_t out_channels = output_shape[1];
  const index_t out_height = output_shape[2];
  const index_t out_width = output_shape[3];
  const index_t padded_in_height = in_height + in_pad_size[0] + in_pad_size[1];
  const index_t padded_in_width = in_width + in_pad_size[2] + in_pad_size[3];

  // pad input and transform input
  const bool is_in_padded =
      in_height != padded_in_height || in_width != padded_in_width;
  auto scratch_buffer = context->device()->scratch_buffer();
  const index_t padded_in_size = is_in_padded ? PadAlignSize(
      sizeof(T) * batch * in_channels * padded_in_height
          * padded_in_width) : 0;
  const index_t pack_filter_size =
      PadAlignSize(sizeof(T) * out_channels * in_channels);
  const index_t pack_input_size =
      PadAlignSize(
          sizeof(T) * in_channels * padded_in_height * padded_in_width);
  const index_t pack_output_size =
      PadAlignSize(
          sizeof(T) * out_channels * padded_in_height * padded_in_width);

  const index_t gemm_pack_size =
      pack_filter_size + pack_input_size + pack_output_size;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(padded_in_size + gemm_pack_size);

  const Tensor *padded_in = input;
  Tensor tmp_padded_in
      (scratch_buffer->Scratch(padded_in_size), DataType::DT_FLOAT);
  if (is_in_padded) {
    tmp_padded_in.Resize({batch, in_channels, padded_in_height,
                          padded_in_width});
    PadInput(*input, in_pad_size[0], in_pad_size[2], &tmp_padded_in);
    padded_in = &tmp_padded_in;
  }

  return gemm_.Compute(context,
                       filter,
                       padded_in,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       out_height * out_width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

void RegisterConv2dK1x1Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Conv2dK1x1<float>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, DeviceType::CPU,
                            float, ImplType::NEON, K1x1));

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Conv2dK1x1<BFloat16>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, DeviceType::CPU,
                            BFloat16, ImplType::NEON, K1x1));
  MACE_REGISTER_FP16_DELEGATOR(
      registry, Conv2dK1x1<float16_t>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, DeviceType::CPU,
                            float16_t, ImplType::NEON, K1x1));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
