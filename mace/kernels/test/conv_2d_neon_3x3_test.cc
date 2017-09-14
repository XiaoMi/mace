//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {

TEST(Conv2dNeon3X3Test, Correctness) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 16;
  index_t channels = 3 + rand() % 100;
  index_t height = 10 + rand() % 100;
  index_t width = 10 + rand() % 100;
  index_t output_channels = 3 + rand() % 100;

  index_t input_size = batch * channels * height * width;
  index_t filter_size = output_channels * channels * 3 * 3;
  std::vector<float> input(input_size, 0.0);
  const index_t input_shape[] = {batch, channels, height, width};
  std::vector<float> filter(filter_size, 0.0);
  const index_t filter_shape[] = {output_channels, channels, 3, 3};
  std::vector<float> bias(output_channels, 0.0);
  const int dilations[] = {1, 1};
  const int strides[] = {1, 1};

  // declare output
  vector<index_t> output_shape;
  vector<int> padding_size;
  kernels::CalcPaddingAndOutputSize(input_shape, filter_shape, dilations, strides, VALID,
                           &output_shape, &padding_size);

  const index_t output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
  std::unique_ptr<float[]> output(new float[output_size]);
  std::unique_ptr<float[]> output_neon(new float[output_size]);


  for (int i = 0; i < input_size; ++i) {
    input[i] = nd(gen);
  }
  for (int i = 0; i < filter_size; ++i) {
    filter[i] = nd(gen);
  }
  for (int i = 0; i < output_channels; ++i) {
    bias[i] = nd(gen);
  }

  kernels::Conv2dFunctor<DeviceType::CPU, float>(strides, padding_size.data(), dilations)(
          input.data(),
          input_shape,
          filter.data(),
          filter_shape,
          bias.data(),
          output.get(),
          output_shape.data()
  );

  kernels::Conv2dFunctor<DeviceType::NEON, float>(strides, padding_size.data(), dilations)(
          input.data(),
          input_shape,
          filter.data(),
          filter_shape,
          bias.data(),
          output_neon.get(),
          output_shape.data()
  );


  for (index_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(output[i], output_neon[i], 1e-3);
  }
}

} //  namespace mace