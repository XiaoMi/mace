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
#include "micro/ops/softmax.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class SoftmaxOpTest : public ::testing::Test {};

namespace {
void Simple(bool use_log = false) {
  const float input[8] = {1, 1, 1, 1, 1, 2, 3, 4};
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {1, 1, 2, 4};
  float output[8] = {0};
  const int32_t output_dim_size = 4;
  int32_t output_dims[output_dim_size] = {0};
  const int32_t expect_dims[output_dim_size] = {1, 1, 2, 4};
  float expected_data1[8] = {-1.3862944, -1.3862944, -1.3862944, -1.3862944,
                             -3.4401896, -2.4401896, -1.4401897, -0.44018975};
  float expected_data2[8] = {0.25, 0.25, 0.25, 0.25,
                             0.0320586, 0.08714432, 0.23688282, 0.6439142};
  float *expect = use_log ? expected_data1 : expected_data2;

  SoftmaxOp softmax_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddArg("use_log", static_cast<int>(use_log))
      .AddOutput(output, output_dims, output_dim_size);

  softmax_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  softmax_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size, 1e-5);
}

}  // namespace

TEST_F(SoftmaxOpTest, CPUSimple) { Simple(); }
TEST_F(SoftmaxOpTest, CPUSimpleUseLog) { Simple(true); }

}  // namespace test
}  // namespace ops
}  // namespace micro
