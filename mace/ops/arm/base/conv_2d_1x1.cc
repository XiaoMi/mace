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

#include <memory>
#include <vector>

#include "mace/core/runtime/runtime.h"

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
  MaceStatus ret = MaceStatus::MACE_RUNTIME_ERROR;
  if (in_height != padded_in_height || in_width != padded_in_width) {
    Runtime *runtime = context->runtime();
    auto mem_type = input->memory_type();
    auto tensor_shape = {batch, in_channels, padded_in_height, padded_in_width};
    std::unique_ptr<Tensor> padded_in =
        make_unique<Tensor>(runtime, DT_FLOAT, mem_type, tensor_shape);
    Tensor tmp_padded_in(runtime, DT_FLOAT, mem_type, tensor_shape);

    PadInput(*input, in_pad_size[0], in_pad_size[2], &tmp_padded_in);

    ret = gemm_.Compute(context, filter, &tmp_padded_in,
                        batch, out_channels, in_channels, in_channels,
                        out_height * out_width, false, false, false,
                        false, true, output);
  } else {
    ret = gemm_.Compute(context, filter, input, batch, out_channels,
                        in_channels, in_channels, out_height * out_width,
                        false, false, false, false, true, output);
  }

  return ret;
}

void RegisterConv2dK1x1Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Conv2dK1x1<float>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU,
                            float, ImplType::NEON, K1x1));

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Conv2dK1x1<BFloat16>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU,
                            BFloat16, ImplType::NEON, K1x1));
  MACE_REGISTER_FP16_DELEGATOR(
      registry, Conv2dK1x1<float16_t>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, RuntimeType::RT_CPU,
                            float16_t, ImplType::NEON, K1x1));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
