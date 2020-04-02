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
#include "micro/ops/expand_dims.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ExpandDimsOpTest : public ::testing::Test {};

namespace {

void ExpandDimsSimpleA() {
  MACE_DEFINE_RANDOM_INPUT(float, input, 6);
  int32_t input_dims[3] = {3, 2, 1};

  float output[6] = {0};
  int32_t output_dims[4] = {0};
  float *expect = input;
  int32_t expect_dims[4] = {3, 1, 2, 1};

  ExpandDimsOp expand_dims_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 3)
      .AddArg("axis", 1)
      .AddOutput(output, output_dims, 4);

  expand_dims_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  expand_dims_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

void ExpandDimsSimpleB() {
  MACE_DEFINE_RANDOM_INPUT(float, input, 6);
  int32_t input_dims[3] = {1, 2, 3};

  float output[6] = {0};
  int32_t output_dims[4] = {0};
  float *expect = input;
  int32_t expect_dims[4] = {1, 2, 3, 1};

  ExpandDimsOp expand_dims_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 3)
      .AddArg("axis", -1)
      .AddOutput(output, output_dims, 4);

  expand_dims_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  expand_dims_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

}  // namespace

TEST_F(ExpandDimsOpTest, ExpandDimsSimple) {
  ExpandDimsSimpleA();
  ExpandDimsSimpleB();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
