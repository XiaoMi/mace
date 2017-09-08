//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <random>
#include <cmath>
#include "gtest/gtest.h"
#include "mace/kernels/addn.h"

using namespace mace;
using namespace mace::kernels;

TEST(NeonTest, AddN) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  int64_t count = 100000;
  vector<float> input1(count);
  vector<float> input2(count);
  vector<float> input3(count);
  vector<float> output(count);
  vector<float> output_neon(count);

  for (int64_t i = 0; i < count; ++i) {
    input1[i] = nd(gen);
    input2[i] = nd(gen);
    input3[i] = nd(gen);
  }

  vector<const float*> inputs { input1.data(), input2.data(), input3.data() };

  AddNFunctor<DeviceType::CPU, float> addn_functor;
  AddNFunctor<DeviceType::NEON, float> neon_addn_functor;
  addn_functor(inputs, &output[0], count);
  neon_addn_functor(inputs, &output_neon[0], count);

  for (int64_t i = 0; i < count; ++i) {
    ASSERT_FLOAT_EQ(output[i], output_neon[i]);
  }
}

