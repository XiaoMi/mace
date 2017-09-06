//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <random>
#include <cmath>
#include "gtest/gtest.h"
#include "mace/kernels/relu.h"

using namespace mace;
using namespace mace::kernels;

TEST(NeonTest, Relu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  int64_t count = 100000;
  vector<float> input(count);
  vector<float> output(count);
  vector<float> output_neon(count);

  for (int64_t i = 0; i < count; ++i) {
    input[i] = nd(gen);
  }

  ReluFunctor<DeviceType::CPU, float> relu_functor;
  ReluFunctor<DeviceType::NEON, float> neon_relu_functor;

  relu_functor(&input[0], &output[0], count);
  neon_relu_functor(&input[0], &output_neon[0], count);

  for (int64_t i = 0; i < count; ++i) {
    ASSERT_FLOAT_EQ(output[i], output_neon[i]);
  }
}

