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
#include "micro/ops/nhwc/batch_norm.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class BatchNormOpTest : public ::testing::Test {};

namespace {

void TestBatchNormOp() {
  float input[12] = {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15};
  int32_t input_dims[4] = {1, 6, 2, 1};
  float scale[1] = {4.0f};
  int32_t scale_dims[1] = {1};
  float offset[1] = {2.0f};
  int32_t offset_dims[1] = {1};
  float mean[1] = {10};
  int32_t mean_dims[1] = {1};
  float var[1] = {11.67f};
  int32_t var_dims[1] = {1};

  float output[12] = {0};
  int32_t output_dims[4] = {0};
  float expect[12] = {-3.8543, -3.8543, -1.5125, -1.5125, 0.8291, 0.8291,
                      3.1708, 3.1708, 5.5125, 5.5125, 7.8543, 7.8543};
  int32_t expect_dims[4] = {1, 6, 2, 1};

  BatchNormOp batch_norm_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(scale, scale_dims, 1)
      .AddInput(offset, offset_dims, 1)
      .AddInput(mean, mean_dims, 1)
      .AddInput(var, var_dims, 1)
      .AddArg("epsilon", 1e-3)
      .AddOutput(output, output_dims, 4);

  batch_norm_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  batch_norm_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-4);
}

}  // namespace

TEST_F(BatchNormOpTest, TestBatchNorm) {
  TestBatchNormOp();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
