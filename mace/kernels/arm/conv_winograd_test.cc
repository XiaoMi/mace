//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include <gtest/gtest.h>
#include <random>
#include <algorithm>

#include "mace/kernels/arm/conv_winograd.h"
#include "mace/core/types.h"

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
  index_t input_size = batch * in_channels * in_height * out_height;
  index_t filter_size = 3 * 3 * in_channels * out_channels;
  index_t output_size = batch * out_channels * out_height * out_width;

  float *input_data = new float[input_size];
  float *filter_data = new float[filter_size];
  float *output_data = new float[output_size];
  float *output_data_ref = new float[output_size];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  std::generate(input_data, input_data + input_size,
                [&gen, &nd] {
                  return std::max(-1.0f, std::min(1.0f, nd(gen)));
                });
  std::generate(filter_data, filter_data + filter_size,
                [&gen, &nd] {
                  return std::max(-1.0f, std::min(1.0f, nd(gen)));
                });

  kernels::ConvRef3x3s1(input_data,
                        filter_data,
                        batch,
                        in_height,
                        in_width,
                        in_channels,
                        out_channels,
                        output_data_ref);

  kernels::WinoGradConv3x3s1(input_data,
                             filter_data,
                             batch,
                             in_height,
                             in_width,
                             in_channels,
                             out_channels,
                             output_data);

  // test
  for (index_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(output_data_ref[i], output_data[i], 0.1);
  }

  delete[]input_data;
  delete[]filter_data;
  delete[]output_data;
  delete[]output_data_ref;
}

}  // namespace kernels
}  // namespace mace
