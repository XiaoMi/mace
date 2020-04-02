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
#include "micro/ops/gtest_utils.h"
#include "micro/ops/nhwc/conv_2d_ref.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class Conv2dOpTest : public ::testing::Test {};

namespace {

void TestNHWCSimple3x3VALID() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {18.1f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3SAME() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[9] = {0};
  int32_t output_dims[4] = {0};
  float expect[9] = {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 3, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3NeqStride() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[6] = {0};
  int32_t output_dims[4] = {0};
  float expect[6] = {8.1f, 8.1f, 12.1f, 12.1f, 8.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 2, 1};

  const int32_t strides[] = {1, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3WithoutBias() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {18.0f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCCombined3x3() {
  float input[50] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 5, 5, 2};
  float filter[36] =
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
       0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int32_t filter_dims[4] = {2, 3, 3, 2};
  float bias[2] = {0.1f, 0.2f};
  int32_t bias_dims[1] = {2};

  float output[18] = {0};
  int32_t output_dims[4] = {0};
  float expect[18] = {8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 18.1f,
                      9.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f};
  int32_t expect_dims[4] = {1, 3, 3, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestFusedNHWCSimple3x3VALID(bool need_bias) {
  float input[18] =
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {-0.1f};
  int32_t bias_dims[1] = {1};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {0.0f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};
  const char activation[] = "RELU";

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddRepeatArg("activation", activation, sizeof(activation))
      .AddOutput(output, output_dims, 4);
  if (need_bias) {
    substitude_op.AddInput(bias, bias_dims, 1);
  }

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestConv1x1() {
  float input[150] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 10, 5};
  float filter[10] =
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  int32_t filter_dims[4] = {2, 1, 1, 5};
  float bias[2] = {0.1f, 0.2f};
  int32_t bias_dims[1] = {2};

  float output[60] = {0};
  int32_t output_dims[4] = {0};
  float expect[60] = {
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f};
  int32_t expect_dims[4] = {1, 3, 10, 2};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

}  // namespace

TEST_F(Conv2dOpTest, TestConv2dVALID) {
  TestNHWCSimple3x3VALID();
}

TEST_F(Conv2dOpTest, TestConv2dSAME) {
  TestNHWCSimple3x3SAME();
}

TEST_F(Conv2dOpTest, NotEqualStrideSimple) {
  TestNHWCSimple3x3NeqStride();
}

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestNHWCSimple3x3WithoutBias();
}

TEST_F(Conv2dOpTest, CPUStride2) {
  TestNHWCCombined3x3();
}

TEST_F(Conv2dOpTest, FusedCPUSimple) {
  TestFusedNHWCSimple3x3VALID(true);
  TestFusedNHWCSimple3x3VALID(false);
}

TEST_F(Conv2dOpTest, CPUConv1x1) {
  TestConv1x1();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
