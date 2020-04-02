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
#include "micro/ops/shape.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ShapeOpTest : public ::testing::Test {};

namespace {

template<typename EXP_TYPE, typename RES_TYPE>
void TestShapeOp(
    const EXP_TYPE *x, const int32_t *x_dims, const uint32_t x_dim_size,
    RES_TYPE *y, int32_t *y_dims, const uint32_t y_dim_size,
    const RES_TYPE *e, const int32_t *e_dims, const uint32_t e_dim_size) {

  ShapeOp shape_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x, x_dims, x_dim_size)
      .AddOutput(y, y_dims, y_dim_size);

  shape_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  shape_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, e, e_dims, e_dim_size);
}

}  // namespace

TEST_F(ShapeOpTest, TestShape) {
  MACE_DEFINE_RANDOM_INPUT(float, x, 6);
  int32_t x_dims[3] = {1, 2, 3};
  int32_t y[3] = {0};
  int32_t y_dims[1] = {0};
  int32_t e[3] = {1, 2, 3};
  int32_t e_dims[1] = {3};

  TestShapeOp(x, x_dims, 3, y, y_dims, 1, e, e_dims, 1);
}

}  // namespace test
}  // namespace ops
}  // namespace micro
