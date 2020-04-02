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
#include "micro/ops/nhwc/conv_2d_c2_s4.h"
#include "micro/ops/nhwc/conv_2d_c3_s4.h"
#include "micro/ops/nhwc/conv_2d_c4_s4.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class Conv2dOptOpTest : public ::testing::Test {};

namespace {

void TestNHWCMulti3x3SAME() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[72] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {4, 3, 3, 2};
  float bias[4] = {0.1f, 0.1f, 0.1f, 0.1f};
  int32_t bias_dims[1] = {4};

  float output[36] = {0};
  int32_t output_dims[4] = {0};
  float expect[36] = {8.1f, 8.1f, 8.1f, 8.1f,
                      12.1f, 12.1f, 12.1f, 12.1f,
                      8.1f, 8.1f, 8.1f, 8.1f,
                      12.1f, 12.1f, 12.1f, 12.1f,
                      18.1f, 18.1f, 18.1f, 18.1f,
                      12.1f, 12.1f, 12.1f, 12.1f,
                      8.1f, 8.1f, 8.1f, 8.1f,
                      12.1f, 12.1f, 12.1f, 12.1f,
                      8.1f, 8.1f, 8.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 3, 4};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dC4S4Op conv_2d_op;
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

void TestNHWCMulti3x3NeqStride() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[36] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {2, 3, 3, 2};
  float bias[2] = {0.1f, 0.1f};
  int32_t bias_dims[1] = {2};

  float output[12] = {0};
  int32_t output_dims[4] = {0};
  float expect[12] = {
      8.1f, 8.1f, 8.1f, 8.1f, 12.1f, 12.1f,
      12.1f, 12.1f, 8.1f, 8.1f, 8.1f, 8.1f
  };
  int32_t expect_dims[4] = {1, 3, 2, 2};

  const int32_t strides[] = {1, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dC2S4Op conv_2d_op;
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

void TestNHWC3Multi3x3NeqStride() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[54] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {3, 3, 3, 2};
  float bias[3] = {0.1f, 0.1f, 0.1f};
  int32_t bias_dims[1] = {3};

  float output[12] = {0};
  int32_t output_dims[4] = {0};
  float expect[18] = {8.1f, 8.1f, 8.1f, 8.1f, 8.1f, 8.1f, 12.1f, 12.1f, 12.1f,
                      12.1f, 12.1f, 12.1f, 8.1f, 8.1f, 8.1f, 8.1f, 8.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 2, 3};

  const int32_t strides[] = {1, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dC3S4Op conv_2d_op;
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

  Conv2dC2S4Op conv_2d_op;
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

  Conv2dC2S4Op conv_2d_op;
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

TEST_F(Conv2dOptOpTest, TestConv2dMultiSAME) {
  TestNHWCMulti3x3SAME();
}

TEST_F(Conv2dOptOpTest, CPUStride2) {
  TestNHWCCombined3x3();
}

TEST_F(Conv2dOptOpTest, CPUConv1x1) {
  TestConv1x1();
}

TEST_F(Conv2dOptOpTest, TestNHWC3Multi3x3NeqStride) {
  TestNHWCMulti3x3NeqStride();
  TestNHWC3Multi3x3NeqStride();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
