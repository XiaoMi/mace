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
#include "micro/ops/reduce.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ReduceOpTest : public ::testing::Test {};

namespace {
typedef ReduceOpBase::ReduceType ReduceType;

void Simple(
    const float *input, const int32_t *input_dims,
    const int32_t input_dim_size,
    const int32_t *axis, const int32_t axis_size,
    float *output, int32_t *output_dims, const int32_t output_dim_size,
    const float *expect, const int32_t *expect_dims,
    ReduceType type, const bool keepdims = true) {
  ReduceOp<float> reduce_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddRepeatArg("axis", axis, axis_size)
      .AddArg("keepdims", keepdims ? 1 : 0)
      .AddArg("reduce_type", static_cast<int32_t>(type))
      .AddOutput(output, output_dims, output_dim_size);

  reduce_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  reduce_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size, 1e-5, 1e-3);
}

void SimpleMean12Test() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 2;
  const int32_t axis[axis_size] = {1, 2};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 1, 4};
  const float expect[8] = {10, 11, 12, 13, 10, 11, 12, 13};
  int32_t output_dims[output_dim_size] = {0};
  float output[8] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MEAN);
}

void SimpleMin12Test() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 2;
  const int32_t axis[axis_size] = {1, 2};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 1, 4};
  const float expect[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  int32_t output_dims[output_dim_size] = {0};
  float output[8] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MIN);
}

void SimpleMax12Test() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 2;
  const int32_t axis[axis_size] = {1, 2};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 1, 4};
  const float expect[8] = {20, 21, 22, 23, 20, 21, 22, 23};
  int32_t output_dims[output_dim_size] = {0};
  float output[8] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MAX);
}

void SimpleMean1Axis() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 1;
  const int32_t axis[axis_size] = {1};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 3, 4};
  const float expect[24] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int32_t output_dims[output_dim_size] = {0};
  float output[24] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MEAN);
}

void SimpleMin1Axis() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 1;
  const int32_t axis[axis_size] = {1};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 3, 4};
  const float expect[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t output_dims[output_dim_size] = {0};
  float output[24] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MIN);
}

void SimpleMax1Axis() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 1;
  const int32_t axis[axis_size] = {1};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {2, 1, 3, 4};
  const float expect[24] = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  int32_t output_dims[output_dim_size] = {0};
  float output[24] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MAX);
}

void Simple2Axis() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {1, 2, 3, 4};
  const float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 2;
  const int32_t axis[axis_size] = {0, 1};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {1, 1, 3, 4};
  const float expect[12] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  int32_t output_dims[output_dim_size] = {0};
  float output[12] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MEAN);

  const int32_t input1_dim_size = 3;
  const int32_t input1_dims[input1_dim_size] = {2, 3, 4};
  const int32_t axis1[axis_size] = {1, 2};
  const int32_t output1_dim_size = 3;
  const int32_t expect1_dims[output1_dim_size] = {2, 1, 1};
  const float expect1[2] = {5.5, 17.5};
  int32_t output1_dims[output_dim_size] = {0};
  float output1[2] = {0};
  Simple(input, input1_dims, input1_dim_size, axis1, axis_size,
         output1, output1_dims, output1_dim_size,
         expect1, expect1_dims, ReduceOpBase::MEAN);

  const int32_t axis2[axis_size] = {0, 2};
  const int32_t expect2_dims[output_dim_size] = {1, 2, 1, 4};
  const float expect2[8] = {4, 5, 6, 7, 16, 17, 18, 19};
  Simple(input, input_dims, input_dim_size, axis2, axis_size,
         output, output_dims, output_dim_size,
         expect2, expect2_dims, ReduceOpBase::MEAN);
}

void Simple3Axis() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {1, 2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 3;
  const int32_t axis[axis_size] = {1, 2, 3};
  const int32_t output_dim_size = 4;
  const int32_t expect_dims[output_dim_size] = {1, 1, 1, 1};
  const float expect[1] = {11.5};
  int32_t output_dims[output_dim_size] = {0};
  float output[1] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MEAN);
}

void CPUSimpleReduceDims() {
  const int32_t input_dim_size = 3;
  const int32_t input_dims[input_dim_size] = {2, 3, 4};
  const float input[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t axis_size = 2;
  const int32_t axis[axis_size] = {0, 1};
  const int32_t output_dim_size = 1;
  const int32_t expect_dims[output_dim_size] = {4};
  const float expect[4] = {10, 11, 12, 13};
  int32_t output_dims[output_dim_size] = {0};
  float output[4] = {0};
  Simple(input, input_dims, input_dim_size, axis, axis_size,
         output, output_dims, output_dim_size,
         expect, expect_dims, ReduceOpBase::MEAN, false);
}

}  // namespace

TEST_F(ReduceOpTest, CPUSimple12) {
  SimpleMean12Test();
  SimpleMin12Test();
  SimpleMax12Test();
}


TEST_F(ReduceOpTest, CPUSimple1Axis) {
  SimpleMean1Axis();
  SimpleMin1Axis();
  SimpleMax1Axis();
}

TEST_F(ReduceOpTest, CPUSimple2Axis) {
  Simple2Axis();
}

TEST_F(ReduceOpTest, CPUSimple3Axis) {
  Simple3Axis();
}

TEST_F(ReduceOpTest, CPUSimpleReduceDims) {
  CPUSimpleReduceDims();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
