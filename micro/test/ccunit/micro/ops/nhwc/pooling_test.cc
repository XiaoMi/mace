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
#include "micro/ops/nhwc/pooling_ref.h"
#include "micro/ops/nhwc/pooling_s4.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class PoolingOpTest : public ::testing::Test {};

namespace {

void TestPoolingOpValidMax() {
  float input[32] = {
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  int32_t input_dims[4] = {1, 4, 4, 2};

  float output[8] = {0};
  int32_t output_dims[4] = {0};
  float expect[8] = {5, 21, 7, 23, 13, 29, 15, 31};
  int32_t expect_dims[4] = {1, 2, 2, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpSameMax() {
  float input[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  int32_t input_dims[4] = {1, 3, 3, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {4, 5, 7, 8};
  int32_t expect_dims[4] = {1, 2, 2, 1};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpValidDilation() {
  float input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int32_t input_dims[4] = {1, 4, 4, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {10, 11, 14, 15};
  int32_t expect_dims[4] = {1, 2, 2, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {2, 2};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpValidAvg() {
  float input[32] = {
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  int32_t input_dims[4] = {1, 4, 4, 2};

  float output[8] = {0};
  int32_t output_dims[4] = {0};
  float expect[8] = {2.5, 18.5, 4.5, 20.5, 10.5, 26.5, 12.5, 28.5};
  int32_t expect_dims[4] = {1, 2, 2, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::AVG)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpSameAvg() {
  float input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int32_t input_dims[4] = {1, 2, 8, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {4.5, 6.5, 8.5, 10.5};
  int32_t expect_dims[4] = {1, 1, 4, 1};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddArg("pooling_type", PoolingType::AVG)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

}  // namespace

TEST_F(PoolingOpTest, TestPoolingValidMax) {
  TestPoolingOpValidMax();
}

TEST_F(PoolingOpTest, TestPoolingSameMax) {
  TestPoolingOpSameMax();
}

TEST_F(PoolingOpTest, TestPoolingValidDilation) {
  TestPoolingOpValidDilation();
}

TEST_F(PoolingOpTest, TestPoolingOpValidAvg) {
  TestPoolingOpValidAvg();
}

TEST_F(PoolingOpTest, TestPoolingOpSameAvg) {
  TestPoolingOpSameAvg();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
