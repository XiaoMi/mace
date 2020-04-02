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
#include "micro/ops/squeeze.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class SqueezeOpTest : public ::testing::Test {};

namespace {

void TestSqueeze(
    const float *input, const int32_t *input_dims,
    const int32_t input_dim_size,
    const int32_t *axis,
    const int32_t axis_size,
    float *output, int32_t *output_dims, const int32_t output_dim_size,
    const float *expect, const int32_t *expect_dims) {
  SqueezeOp squeeze_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddOutput(output, output_dims, output_dim_size);
  if (axis != NULL && axis_size > 0) {
    substitude_op.AddRepeatArg("axis", axis, axis_size);
  }

  squeeze_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  squeeze_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size, 1e-5);
}

void TestSqueeze() {
  MACE_DEFINE_RANDOM_INPUT(float, input, 8);
  const int32_t dims1214[] = {1, 2, 1, 4};
  const int32_t dims24[] = {2, 4};
  const int32_t dims124[] = {1, 2, 4};
  const int32_t dims1411[] = {1, 4, 1, 1};
  const int32_t dims141[] = {1, 4, 1};

  float output[8] = {0};
  int32_t output_dims[10] = {0};

  TestSqueeze(input, dims1214, 4, NULL, 0,
              output, output_dims, 2, input, dims24);

  int32_t axis_size = 1;
  int32_t axis[] = {1};
  TestSqueeze(input, dims1214, 4, axis, axis_size,
              output, output_dims, 4, input, dims1214);

  int32_t axis2[] = {2};
  TestSqueeze(input, dims1214, 4, axis2, axis_size,
              output, output_dims, 3, input, dims124);

  MACE_DEFINE_RANDOM_INPUT(float, input3, 4);
  int32_t axis3[2] = {1, 2};
  TestSqueeze(input, dims1411, 4, axis3, 2,
              output, output_dims, 3, input, dims141);
}

}  // namespace

TEST_F(SqueezeOpTest, TestSqueeze) {
  TestSqueeze();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
