//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <random>
#include <cmath>
#include "gtest/gtest.h"
#include "mace/kernels/neon/addn_neon.h"
#include "mace/kernels/addn.h"

using namespace mace;
using namespace mace::kernels;

TEST(NeonTest, AddN) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  int64 count = 100000;
  Tensor input_tensor1(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor1.Resize({100, 1000});
  Tensor input_tensor2(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor2.ResizeLike(input_tensor1);
  Tensor input_tensor3(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor3.ResizeLike(input_tensor1);
  vector<const Tensor*> input_tensors {&input_tensor1,
                                       &input_tensor2,
                                       &input_tensor3};

  Tensor output_tensor(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor.ResizeLike(input_tensors[0]);
  Tensor output_tensor_neon(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor_neon.ResizeLike(input_tensors[0]);

  float *input1 = input_tensor1.mutable_data<float>();
  float *input2 = input_tensor2.mutable_data<float>();
  float *input3 = input_tensor3.mutable_data<float>();
  float *output = output_tensor.mutable_data<float>();
  float *output_neon = output_tensor_neon.mutable_data<float>();

  for (int64 i = 0; i < count; ++i) {
    input1[i] = nd(gen);
    input2[i] = nd(gen);
    input3[i] = nd(gen);
  }

  AddNFuntion<float>(input_tensors, &output_tensor);
  NeonAddNFuntion_float(input_tensors, &output_tensor_neon);

  ASSERT_EQ(count, output_tensor.size());
  ASSERT_EQ(count, output_tensor_neon.size());
  for (int64 i = 0; i < count; ++i) {
    ASSERT_FLOAT_EQ(output[i], output_neon[i]);
  }
}

