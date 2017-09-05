//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <random>
#include <cmath>
#include "gtest/gtest.h"
#include "mace/kernels/neon/relu_neon.h"
#include "mace/kernels/relu.h"

using namespace mace;
using namespace mace::kernels;

TEST(NeonTest, Relu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  int64 count = 100000;
  Tensor input_tensor(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor.Resize({100, 1000});
  Tensor output_tensor(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor.ResizeLike(input_tensor);
  Tensor output_tensor_neon(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor_neon.ResizeLike(input_tensor);

  float *input = input_tensor.mutable_data<float>();
  float *output = output_tensor.mutable_data<float>();
  float *output_neon = output_tensor_neon.mutable_data<float>();

  for (int64 i = 0; i < count; ++i) {
    input[i] = nd(gen);
  }

  ReluFuntion<float>(&input_tensor, &output_tensor);
  NeonReluFuntion_float(&input_tensor, &output_tensor_neon);

  ASSERT_EQ(count, output_tensor.size());
  ASSERT_EQ(count, output_tensor_neon.size());
  for (int64 i = 0; i < count; ++i) {
    ASSERT_FLOAT_EQ(output[i], output_neon[i]);
  }
}

