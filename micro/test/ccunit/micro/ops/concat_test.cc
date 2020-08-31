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

#include "micro/ops/concat.h"
#include "gtest/gtest.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class ConcatOpTest : public ::testing::Test {};

TEST_F(ConcatOpTest, TestValueTypeDouble) {
  // clang-format off
  double x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  double x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  double y[2 * 5] = {};
  int32_t y_dims[2] = {2, 5};
  int32_t y_dim_size = 2;

  double y_g[2 * 5] = {
    0, 1, 2, 6, 7,
    3, 4, 5, 8, 9
  };
  int32_t y_g_dims[2] = {2, 5};
  int32_t y_g_dim_size = 2;
  // clang-format on

  ConcatOp<double> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<double>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestValueTypeFloat) {
  // clang-format off
  float x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  float x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  float y[2 * 5] = {};
  int32_t y_dims[2] = {2, 5};
  int32_t y_dim_size = 2;

  float y_g[2 * 5] = {
    0, 1, 2, 6, 7,
    3, 4, 5, 8, 9
  };
  int32_t y_g_dims[2] = {2, 5};
  int32_t y_g_dim_size = 2;
  // clang-format on

  ConcatOp<float> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<float>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestInputOrder) {
  // clang-format off
  int32_t x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  int32_t x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  int32_t y[2 * 5] = {};
  int32_t y_dims[2] = {2, 5};
  int32_t y_dim_size = 2;

  int32_t y_g[2 * 5] = {
    6, 7, 0, 1, 2,
    8, 9, 3, 4, 5
  };
  int32_t y_g_dims[2] = {2, 5};
  int32_t y_g_dim_size = 2;
  // clang-format on

  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x1, x1_dims, x1_dim_size)
      .AddInput(x0, x0_dims, x0_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestAxis1) {
  // clang-format off
  int32_t x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  int32_t x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  int32_t y[2 * 5] = {};
  int32_t y_dims[2] = {2, 5};
  int32_t y_dim_size = 2;

  int32_t y_g[2 * 5] = {
    0, 1, 2, 6, 7,
    3, 4, 5, 8, 9
  };
  int32_t y_g_dims[2] = {2, 5};
  int32_t y_g_dim_size = 2;
  // clang-format on
  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestAxis0) {
  // clang-format off
  int32_t x0[3 * 2] = {
    0, 1,
    2, 3,
    4, 5
  };
  int32_t x0_dims[2] = { 3, 2};
  int32_t x0_dim_size = 2;

  int32_t x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  int32_t y[5 * 2] = {};
  int32_t y_dims[2] = {5, 2};
  int32_t y_dim_size = 2;

  int32_t y_g[5 * 2] = {
    0, 1,
    2, 3,
    4, 5,
    6, 7,
    8, 9
  };
  int32_t y_g_dims[2] = {5, 2};
  int32_t y_g_dim_size = 2;
  // clang-format on
  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 0)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestInputNumber1) {
  // clang-format off
  int32_t x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  int32_t y[2 * 3] = {};
  int32_t y_dims[2] = {2, 3};
  int32_t y_dim_size = 2;

  int32_t y_g[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t y_g_dims[2] = {2, 3};
  int32_t y_g_dim_size = 2;
  // clang-format on

  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestInputNumber2) {
  // clang-format off
  int32_t x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  int32_t x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  int32_t y[2 * 5] = {};
  int32_t y_dims[2] = {2, 5};
  int32_t y_dim_size = 2;

  int32_t y_g[2 * 5] = {
    0, 1, 2, 6, 7,
    3, 4, 5, 8, 9
  };
  int32_t y_g_dims[2] = {2, 5};
  int32_t y_g_dim_size = 2;
  // clang-format on
  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

TEST_F(ConcatOpTest, TestInputNumber3) {
  // clang-format off
  int32_t x0[2 * 3] = {
    0, 1, 2,
    3, 4, 5
  };
  int32_t x0_dims[2] = { 2, 3};
  int32_t x0_dim_size = 2;

  int32_t x1[2 * 2] = {
    6, 7,
    8, 9
  };
  int32_t x1_dims[2] = { 2, 2};
  int32_t x1_dim_size = 2;

  int32_t y[2 * 7] = {};
  int32_t y_dims[2] = {2, 7};
  int32_t y_dim_size = 2;

  int32_t y_g[2 * 7] = {
    0, 1, 2, 6, 7, 6, 7,
    3, 4, 5, 8, 9, 8, 9
  };
  int32_t y_g_dims[2] = {2, 7};
  int32_t y_g_dim_size = 2;
  // clang-format on

  ConcatOp<int32_t> Concat_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(x0, x0_dims, x0_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddInput(x1, x1_dims, x1_dim_size)
      .AddArg("axis", 1)
      .AddOutput(y, y_dims, y_dim_size);

  Concat_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  Concat_op.Run();

  ExpectTensorNear<int32_t>(y, y_dims, y_dim_size, y_g, y_g_dims, y_g_dim_size);
}

}  // namespace test
}  // namespace ops
}  // namespace micro
