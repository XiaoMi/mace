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

#include "mace/ops/opencl/buffer/conv_2d.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

bool Conv2dKernel::CheckUseWinograd(
    OpenclExecutor *executor,
    const std::vector<index_t> &filter_shape,
    const std::vector<index_t> &output_shape,
    const int *strides,
    const int *dilations,
    int *wino_block_size) {
  MACE_UNUSED(kwg_size_);
  MACE_UNUSED(executor);
  MACE_UNUSED(output_shape);
  MACE_UNUSED(wino_block_size);
  return (filter_shape[2] == 3 && filter_shape[3] == 3 &&
      strides[0] == 1 && strides[1] == 1 &&
      dilations[0] == 1 && dilations[1] == 1);
}

MaceStatus Conv2dKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    const int *strides,
    const Padding &padding_type,
    const std::vector<int> &padding_data,
    const int *dilations,
    const ActivationType activation,
    const float relux_max_limit,
    const float activation_coefficient,
    const int winograd_blk_size,
    Tensor *output) {
  MACE_UNUSED(winograd_blk_size);
  StatsFuture pad_future, conv_future;
  index_t filter_h = filter->dim(2);
  index_t filter_w = filter->dim(3);
  // Reshape output
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), filter->shape().data(),
                   padding_data.data(), dilations, strides, RoundType::FLOOR,
                   output_shape.data());
  }

  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  // calculate padded input shape
  index_t width = output_shape[2];
  index_t channels = output_shape[3];

  index_t input_height = input->dim(1);
  index_t input_width = input->dim(2);
  index_t input_channels = input->dim(3);

  int pad_top = paddings[0] >> 1;
  int pad_left = paddings[1] >> 1;

  MACE_CHECK(filter->dim(0) == channels, filter->dim(0), " != ", channels);
  MACE_CHECK(filter->dim(1) == input_channels, filter->dim(1), " != ",
             input_channels);

  std::function<MaceStatus(const Tensor *input, Tensor *output)> conv_func;

  // Mark whether input changed or not
  bool input_changed = IsResetArgsNeeded(context, input_shape_, input->shape());
  input_shape_ = input->shape();

  bool use_1x1 = filter_h == 1 && filter_w == 1;

  std::vector<index_t> padded_output_shape = output_shape;
  index_t tile_w, tile_c = 4;
  if (use_1x1) {
    tile_w = 2;
  } else {
    tile_w = 4;
  }
  padded_output_shape[2] = RoundUp<index_t>(width, tile_w);

  std::vector<index_t> padded_input_shape = input->shape();
  padded_input_shape[1] = input_height + paddings[0];
  padded_input_shape[2] = (padded_output_shape[2] - 1) * strides[1] +
      (filter_w - 1) * dilations[1] + 1;
  padded_input_shape[3] = RoundUp<index_t>(input_channels, tile_c);

  const Tensor *padded_input_ptr = input;
  // pad input
  std::unique_ptr<Tensor> padded_input;
  if (padded_input_shape[1] != input_height ||
      padded_input_shape[2] != input_width ||
      padded_input_shape[3] != input_channels) {
    // decide scratch size before allocate it
    index_t padded_input_size =
        std::accumulate(padded_input_shape.begin(),
                        padded_input_shape.end(),
                        1, std::multiplies<index_t>()) +
            MACE_EXTRA_BUFFER_PAD_SIZE / GetEnumTypeSize(input->dtype());
    auto *runtime = context->runtime();
    padded_input.reset(new Tensor(
        runtime, input->dtype(), output->memory_type(), {padded_input_size}));
    runtime->AllocateBufferForTensor(padded_input.get(), RENT_SCRATCH);
    padded_input->Resize(padded_input_shape);
    PadInput(context, &kernels_[0], input, pad_top, pad_left,
             input_changed, padded_input.get(), &pad_future);
    padded_input_ptr = padded_input.get();
  }

  if (use_1x1) {
    conv_func = [&](const Tensor *pad_input, Tensor *output) -> MaceStatus {
      return conv2d::Conv2d1x1(
          context, &kernels_[1], pad_input, filter, bias, strides,
          activation, relux_max_limit,
          activation_coefficient, input_changed, output, &conv_future);
    };
  } else {
    conv_func = [&](const Tensor *pad_input, Tensor *output) -> MaceStatus {
      return conv2d::Conv2dGeneral(
          context, &kernels_[1], pad_input, filter, bias, strides, dilations,
          activation, relux_max_limit,
          activation_coefficient, input_changed, output, &conv_future);
    };
  }
  MACE_RETURN_IF_ERROR(conv_func(padded_input_ptr, output));
  MergeMultipleFutureWaitFn({pad_future, conv_future}, context->future());

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
