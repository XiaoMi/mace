//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/kernels/conv_2d.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static constexpr index_t kInputChannelBlockSize = 2;
static constexpr index_t kOutputChannelBlockSize = 4;

extern void Conv2dOpenclK1x1S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, Tensor *output) {
  const index_t batch = output->shape()[0];
  const index_t channels = output->shape()[1];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];

  const index_t input_batch = input->shape()[0];
  const index_t input_channels = input->shape()[1];
  const index_t input_height = input->shape()[2];
  const index_t input_width = input->shape()[3];

  MACE_CHECK(input_batch == batch && input_height == height &&
             input_width == width);

  const index_t total_pixels = height * width;
  const index_t round_up_channels = RoundUp(channels, kOutputChannelBlockSize);

};

}  // namespace kernels
}  // namespace mace
