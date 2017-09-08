//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <random>
#include "gtest/gtest.h"
#include "mace/kernels/batch_norm.h"

namespace mace {

TEST(BatchNormNeonTest, Simple) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  srand(time(NULL));

  // generate random input
  TIndex batch = 1 + rand() % 128;
  TIndex channels = 3;
  TIndex height = 2 + rand() % 100;
  TIndex width = 2 + rand() % 100;

  TIndex input_size = batch * channels * height * width;
  std::vector<float> input(input_size, 0.0);
  std::vector<float> scale(channels, 0.0);
  std::vector<float> offset(channels, 0.0);
  std::vector<float> mean(channels, 0.0);
  std::vector<float> var(channels, 0.0);

  for (int i = 0; i < input_size; ++i) {
    input[i] = nd(gen);
  }
  for (int i = 0; i < channels; ++i) {
    scale[i] = nd(gen);
    offset[i] = nd(gen);
    mean[i] = nd(gen);
    var[i] = std::abs(nd(gen));
  }

  // declare output
  std::unique_ptr<float[]> output(new float[input_size]);
  std::unique_ptr<float[]> output_neon(new float[input_size]);

  kernels::BatchNormFunctor<DeviceType::CPU, float>(1e-5)(
          input.data(),
          scale.data(),
          offset.data(),
          mean.data(),
          var.data(),
          batch,
          channels,
          height * width,
          output.get()
  );
  kernels::BatchNormFunctor<DeviceType::NEON, float>(1e-5)(
          input.data(),
          scale.data(),
          offset.data(),
          mean.data(),
          var.data(),
          batch,
          channels,
          height * width,
          output_neon.get()
  );

  for (TIndex i = 0; i < input_size; ++i) {
    EXPECT_FLOAT_EQ(output[i], output_neon[i]);
  }

}

} //  namespace mace