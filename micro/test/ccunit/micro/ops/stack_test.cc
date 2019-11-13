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
#include "micro/ops/stack.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class StackOpTest : public ::testing::Test {};

namespace {

void TestStack(
    const float **inputs, const int32_t inputs_size, const int32_t *input_dims,
    const int32_t input_dim_size, int axis,
    float *output, int32_t *output_dims, const int32_t output_dim_size,
    const float *expect, const int32_t *expect_dims) {
  StackOp<float> stack_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddArg("axis", axis)
      .AddOutput(output, output_dims, output_dim_size);
  for (int32_t i = 0; i < inputs_size; ++i) {
    substitude_op.AddInput(inputs[i], input_dims, input_dim_size);
  }

  stack_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  stack_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size, 1e-5);
}

void TestStackScalar() {
  const float input0[1] = {1};
  const float input1[1] = {2};
  const float input2[1] = {3};
  const int32_t axis = 0;

  float output[3] = {0};
  const int32_t output_dim_size = 1;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[3] = {1, 2, 3};
  const int32_t expect_dims[output_dim_size] = {3};

  const float *inputs[] = {input0, input1, input2};
  TestStack(inputs, 3, NULL, 0, axis,
            output, output_dims, output_dim_size, expect, expect_dims);
}

void TestStackVector() {
  const float input0[] = {1, 4};
  const float input1[] = {2, 5};
  const float input2[] = {3, 6};
  const int32_t input_dim_size = 1;
  const int32_t input_dims[input_dim_size] = {2};
  int32_t axis = 0;

  float output[6] = {0};
  const int32_t output_dim_size = 2;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[6] = {1, 4, 2, 5, 3, 6};
  const int32_t expect_dims[output_dim_size] = {3, 2};

  const float *inputs[] = {input0, input1, input2};
  TestStack(inputs, 3, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect, expect_dims);

  axis = -2;
  TestStack(inputs, 3, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect, expect_dims);

  axis = -1;
  const float expect2[6] = {1, 2, 3, 4, 5, 6};
  const int32_t expect_dims2[output_dim_size] = {2, 3};
  TestStack(inputs, 3, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect2, expect_dims2);
}

void TestStackHighRank() {
  const float input0[] = {1, 2, 3, 4, 5, 6};
  const float input1[] = {7, 8, 9, 10, 11, 12};
  const int32_t input_dim_size = 2;
  const int32_t input_dims[input_dim_size] = {2, 3};
  int32_t axis = -3;

  float output[12] = {0};
  const int32_t output_dim_size = 3;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int32_t expect_dims[output_dim_size] = {2, 2, 3};

  const float *inputs[] = {input0, input1};
  TestStack(inputs, 2, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect, expect_dims);

  axis = 1;
  const float expect1[12] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  TestStack(inputs, 2, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect1, expect_dims);

  axis = 2;
  const int32_t expect_dims2[output_dim_size] = {2, 3, 2};
  const float expect2[12] = {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
  TestStack(inputs, 2, input_dims, input_dim_size, axis,
            output, output_dims, output_dim_size, expect2, expect_dims2);
}
}  // namespace

TEST_F(StackOpTest, TestStackScalar) {
  TestStackScalar();
}

TEST_F(StackOpTest, TestStackVector) {
  TestStackVector();
}

TEST_F(StackOpTest, TestStackHighRank) {
  TestStackHighRank();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
