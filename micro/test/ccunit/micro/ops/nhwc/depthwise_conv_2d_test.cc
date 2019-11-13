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
#include "micro/ops/nhwc/depthwise_conv_2d_ref.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class DepthwiseConv2dOpTest : public ::testing::Test {};

namespace {
void SimpleValidTest() {
  float input[18] = {1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12, 7, 14, 8, 16, 9, 18};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[8] = {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f, 8.0f};
  int32_t filter_dims[4] = {1, 2, 2, 2};
  float bias[2] = {0.1f, 0.2f};
  int32_t bias_dims[1] = {2};

  float output[8] = {0};
  int32_t output_dims[4] = {0};
  float expect[8] = {37.1f, 148.2f, 47.1f, 188.2f,
                     67.1f, 268.2f, 77.1f, 308.2f};
  int32_t expect_dims[4] = {1, 2, 2, 2};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  DepthwiseConv2dRefOp depthwise_conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  depthwise_conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  depthwise_conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims,
                          4, expect, expect_dims, 4, 1e-5);
}

void MultiC2ValidTest() {
  float input[18] = {1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12, 7, 14, 8, 16, 9, 18};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[16] = {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f, 8.0f,
                      1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f, 8.0f};
  int32_t filter_dims[4] = {2, 2, 2, 2};
  float bias[4] = {0.1f, 0.1f, 0.2f, 0.2f};
  int32_t bias_dims[1] = {4};

  float output[16] = {0};
  int32_t output_dims[4] = {0};
  float expect[16] = {
      37.1f, 37.1f, 148.2f, 148.2f, 47.1f, 47.1f, 188.2f, 188.2f,
      67.1f, 67.1f, 268.2f, 268.2f, 77.1f, 77.1f, 308.2f, 308.2f
  };
  int32_t expect_dims[4] = {1, 2, 2, 4};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  DepthwiseConv2dRefOp depthwise_conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  depthwise_conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  depthwise_conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

}  // namespace

TEST_F(DepthwiseConv2dOpTest, SimpleCPU) {
  SimpleValidTest();
}

TEST_F(DepthwiseConv2dOpTest, MuiltiC2CPU) {
  MultiC2ValidTest();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
