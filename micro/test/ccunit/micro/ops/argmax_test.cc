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
#include "micro/ops/argmax.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ArgMaxOpTest : public ::testing::Test {};

namespace {

void ArgMaxTest(
    const float *input, const int32_t *input_dims,
    const int32_t input_dim_size,
    int32_t *output, int32_t *output_dims, const int32_t output_dim_size,
    const int32_t *expect, const int32_t *expect_dims) {
  ArgMaxOp<float> argmax_op;
  int32_t axis[] = {-1};
  int32_t axis_dims[1] = {1};
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddInput(axis, axis_dims, 0)
      .AddOutput(output, output_dims, output_dim_size);

  argmax_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  argmax_op.Run();

  ExpectTensorNear<int32_t>(output, output_dims, output_dim_size,
                            expect, expect_dims, output_dim_size, 1e-5, 1e-3);
}

void ArgMaxTextVector() {
  const float input[3] = {-3, -1, -2};
  const int32_t input_dims[1] = {3};

  int32_t output[1] = {0};
  int32_t output_dims[1] = {0};

  const int32_t expect[1] = {1};
  const int32_t expect_dims[1] = {0};

  ArgMaxTest(input, input_dims, 1,
             output, output_dims, 0,
             expect, expect_dims);
}

void ArgMaxTextMatrix() {
  const float input[9] = {4, 5, 6, 9, 8, 7, 1, 2, 3};
  const int32_t input_dims[2] = {3, 3};

  int32_t output[3] = {0};
  int32_t output_dims[1] = {0};

  const int32_t expect[3] = {2, 0, 2};
  const int32_t expect_dims[1] = {3};

  ArgMaxTest(input, input_dims, 1,
             output, output_dims, 1,
             expect, expect_dims);
}

void ArgMaxTextHighRank() {
  const float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const int32_t input_dims[4] = {1, 2, 2, 3};

  int32_t output[4] = {0};
  int32_t output_dims[3] = {0};

  const int32_t expect[4] = {2, 2, 2, 2};
  const int32_t expect_dims[3] = {1, 2, 2};

  ArgMaxTest(input, input_dims, 4,
             output, output_dims, 3,
             expect, expect_dims);
}

}  // namespace

TEST_F(ArgMaxOpTest, Vector) {
  ArgMaxTextVector();
}

TEST_F(ArgMaxOpTest, Matrix) {
  ArgMaxTextMatrix();
}

TEST_F(ArgMaxOpTest, HighRank) {
  ArgMaxTextHighRank();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
