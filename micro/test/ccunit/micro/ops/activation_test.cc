// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "micro/ops/activation.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ActivationOpTest : public ::testing::Test {};
namespace {

void TestSimpleRelu() {
  float input[16] = {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] = {0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "RELU";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestSimpleLeakyRelu() {
  float input[16] = {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] =
      {-0.7, 7, -0.6, 6, -0.5, 5, -0.4, 4, -0.3, 3, -0.2, 2, -0.1, 1, 0, 0};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "LEAKYRELU";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddArg("leakyrelu_coefficient", 0.1f)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestUnalignedSimpleRelu() {
  float input[6] = {-7, 7, -6, 6, -5, 5};
  int32_t input_dims[4] = {1, 3, 2, 1};

  float output[6] = {0};
  int32_t output_dims[4] = {0};
  float expect[6] = {0, 7, 0, 6, 0, 5};
  int32_t expect_dims[4] = {1, 3, 2, 1};

  const char activation_type[] = "RELU";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestSimpleRelux() {
  float input[16] = {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] = {0, 6, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "RELUX";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddArg("max_limit", 6)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestSimplePrelu() {
  float input[16] = {-7, 7, -6, 6, -5, -5, -4, -4, -3, 3, -2, 2, -1, -1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};
  float alpha[2] = {2.0, 3.0};
  int32_t alpha_dims[1] = {2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] =
      {-14, 7, -12, 6, -10, -15, -8, -12, -6, 3, -4, 2, -2, -3, 0, 0};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "PRELU";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(alpha, alpha_dims, 1)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestSimpleTanh() {
  float input[16] = {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] =
      {-0.99999834, 0.99999834, -0.99998771, 0.99998771, -0.9999092, 0.9999092,
       -0.9993293, 0.9993293, -0.99505475, 0.99505475, -0.96402758, 0.96402758,
       -0.76159416, 0.76159416, 0., 0.};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "TANH";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void TestSimpleSigmoid() {
  float input[16] = {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0};
  int32_t input_dims[4] = {2, 2, 2, 2};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] =
      {9.11051194e-04, 9.99088949e-01, 2.47262316e-03, 9.97527377e-01,
       6.69285092e-03, 9.93307149e-01, 1.79862100e-02, 9.82013790e-01,
       4.74258732e-02, 9.52574127e-01, 1.19202922e-01, 8.80797078e-01,
       2.68941421e-01, 7.31058579e-01, 5.00000000e-01, 5.00000000e-01};
  int32_t expect_dims[4] = {2, 2, 2, 2};

  const char activation_type[] = "SIGMOID";
  const uint32_t arg_type_len = sizeof(activation_type);

  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);

  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  activation_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

}  // namespace

TEST_F(ActivationOpTest, TestSimpleRelu) {
  TestSimpleRelu();
}

TEST_F(ActivationOpTest, TestSimpleLeakyRelu) {
  TestSimpleLeakyRelu();
}

TEST_F(ActivationOpTest, TestUnalignedSimpleRelu) {
  TestUnalignedSimpleRelu();
}

TEST_F(ActivationOpTest, TestSimpleRelux) {
  TestSimpleRelux();
}

TEST_F(ActivationOpTest, TestSimplePrelu) {
  TestSimplePrelu();
}

TEST_F(ActivationOpTest, TestSimpleTanh) {
  TestSimpleTanh();
}

TEST_F(ActivationOpTest, TestSimpleSigmoid) {
  TestSimpleSigmoid();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
