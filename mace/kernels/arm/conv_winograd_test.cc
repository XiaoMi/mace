// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <random>

#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/kernels/arm/conv_winograd.h"

namespace mace {
namespace kernels {

TEST(ConvWinogradTest, winograd) {
  index_t batch = 1;
  index_t in_height = 32;
  index_t in_width = 32;
  index_t in_channels = 64;
  index_t out_channels = 128;

  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;
  index_t input_size = batch * in_channels * in_height * in_width;
  index_t filter_size = 3 * 3 * in_channels * out_channels;
  index_t output_size = batch * out_channels * out_height * out_width;

  Tensor input;
  Tensor filter;
  Tensor output;
  Tensor output_ref;

  input.Resize({batch, in_channels, in_height, in_width});
  filter.Resize({out_channels, in_channels, 3, 3});
  output.Resize({batch, out_channels, out_height, out_width});
  output_ref.Resize({batch, out_channels, out_height, out_width});

  float *input_data = input.mutable_data<float>();
  float *filter_data = filter.mutable_data<float>();
  float *output_data = output.mutable_data<float>();
  float *output_data_ref = output.mutable_data<float>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  std::generate(input_data, input_data + input_size, [&gen, &nd] {
    return std::max(-1.0f, std::min(1.0f, nd(gen)));
  });
  std::generate(filter_data, filter_data + filter_size, [&gen, &nd] {
    return std::max(-1.0f, std::min(1.0f, nd(gen)));
  });

  kernels::ConvRef3x3s1(input_data, filter_data, batch, in_height, in_width,
                        in_channels, out_channels, output_data_ref);

  kernels::WinoGradConv3x3s1(input_data, filter_data, batch, in_height,
                             in_width, in_channels, out_channels, 6,
                             output_data);

  // test
  for (index_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(output_data_ref[i], output_data[i], 0.1) << " with index " << i;
  }
}

}  // namespace kernels
}  // namespace mace
