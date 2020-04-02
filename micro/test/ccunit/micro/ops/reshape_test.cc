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
#include "micro/ops/reshape.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ReshapeOpTest : public ::testing::Test {};

namespace {

template<typename T>
void TestReshapeOp(
    const T *input, const int32_t *input_dims, const uint32_t input_dim_size,
    const int32_t *shape, const int32_t *shape_dims,
    T *y, int32_t *y_dims, const uint32_t y_dim_size,
    const T *e, const int32_t *e_dims, const uint32_t e_dim_size) {

  ReshapeOp reshape_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddInput(shape, shape_dims, 1)
      .AddOutput(y, y_dims, y_dim_size);

  reshape_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  reshape_op.Run();

  ExpectTensorNear<T>(y, y_dims, y_dim_size, e, e_dims, e_dim_size);
}

}  // namespace

TEST_F(ReshapeOpTest, TestReshape) {
  MACE_DEFINE_RANDOM_INPUT(float, x, 6);
  int32_t x_dims[3] = {1, 2, 3};
  int32_t shape[2] = {3, 2};
  int32_t shape_dims[1] = {2};

  float y[6] = {0};
  int32_t y_dims[2] = {0};

  int32_t e_dims[2] = {3, 2};

  TestReshapeOp(x, x_dims, 3, shape, shape_dims,
                y, y_dims, 2, x, e_dims, 2);
}

}  // namespace test
}  // namespace ops
}  // namespace micro
