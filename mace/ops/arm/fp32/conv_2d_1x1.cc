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


#include "mace/ops/arm/fp32/conv_2d_1x1.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Conv2dK1x1::Compute(const OpContext *context,
                               const Tensor *input,
                               const Tensor *filter,
                               Tensor *output) {
  index_t batch = input->dim(0);
  index_t height = input->dim(2);
  index_t width = input->dim(3);
  index_t in_channels = input->dim(1);
  index_t out_channels = filter->dim(0);
  MACE_RETURN_IF_ERROR(output->Resize({batch, out_channels, height, width}));
  context->device()->scratch_buffer()->Rewind();
  return gemm_.Compute(context,
                       filter,
                       input,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       height * width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
