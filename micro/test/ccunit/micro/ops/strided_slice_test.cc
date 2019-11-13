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
#include "micro/ops/strided_slice.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class StridedSliceOpTest : public ::testing::Test {};

namespace {

void TestStridedSlice(
    const float *input, const int32_t *input_dims, const int32_t input_dim_size,
    const int32_t *begin_indices, const int32_t *end_indices,
    const int32_t *strides,
    const int32_t *indices_dims, const int32_t indices_dim_size,
    const int32_t begin_mask, const int32_t end_mask,
    const int32_t ellipsis_mask, const int32_t new_axis_mask,
    const int32_t shrink_axis_mask, const int32_t output_dim_size,
    float *output, int32_t *output_dims,
    const float *expect, const int32_t *expect_dims) {
  StridedSliceOp<float> strided_slice_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddInput(begin_indices, indices_dims, indices_dim_size)
      .AddInput(end_indices, indices_dims, indices_dim_size)
      .AddInput(strides, indices_dims, indices_dim_size)
      .AddArg("begin_mask", begin_mask)
      .AddArg("end_mask", end_mask)
      .AddArg("ellipsis_mask", ellipsis_mask)
      .AddArg("new_axis_mask", new_axis_mask)
      .AddArg("shrink_axis_mask", shrink_axis_mask)
      .AddOutput(output, output_dims, output_dim_size);

  strided_slice_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  strided_slice_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size);
}

void TestSlice(
    const float *input, const int32_t *input_dims, const int32_t input_dim_size,
    const int32_t *begin_indices, const int32_t *indice_sizes,
    const int32_t *indices_dims, const int32_t indices_dim_size,
    float *output, int32_t *output_dims, const int32_t output_dim_size,
    const float *expect, const int32_t *expect_dims) {
  StridedSliceOp<float> strided_slice_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddInput(begin_indices, indices_dims, indices_dim_size)
      .AddInput(indice_sizes, indices_dims, indices_dim_size)
      .AddArg("slice", 1)
      .AddOutput(output, output_dims, output_dim_size);

  strided_slice_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  strided_slice_op.Run();

  ExpectTensorNear<float>(output, output_dims, output_dim_size,
                          expect, expect_dims, output_dim_size);
}

void TestStridedSliceByFirstAxis() {
  const float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int32_t begin_indices[] = {1, 0, 0};
  const int32_t end_indices[] = {2, 3, 2};
  const int32_t strides[] = {1, 1, 1};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {3};
  const int32_t input_dim_size = 3;
  const int32_t input_dims[input_dim_size] = {2, 3, 2};

  float output[6] = {0};
  const int32_t output_dim_size = 3;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[6] = {7, 8, 9, 10, 11, 12};
  const int32_t expect_dims[output_dim_size] = {1, 3, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const int32_t output_dim_size1 = 2;
  int32_t output_dims1[output_dim_size1] = {0};
  const int32_t expect_dims1[output_dim_size1] = {3, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 1, output_dim_size1,
                   output, output_dims1, expect, expect_dims1);

  const int32_t begin_indices2[] = {1, 1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices, strides,
                   indices_dims, indices_dim_size,
                   6, 6, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);
}

void TestStridedSliceRank1() {
  const float input[] = {1, 2, 3, 4};
  const int32_t begin_indices[] = {1};
  const int32_t end_indices[] = {3};
  const int32_t strides[] = {1};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {1};
  const int32_t input_dim_size = 1;
  const int32_t input_dims[input_dim_size] = {4};

  float output[4] = {0};
  const int32_t output_dim_size = 1;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[2] = {2, 3};
  const int32_t expect_dims[output_dim_size] = {2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const int32_t begin_indices1[] = {-3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices1, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const int32_t begin_indices2[] = {-2};
  const int32_t end_indices2[] = {-4};
  const int32_t strides2[] = {-1};
  const float expect2[2] = {3, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices2, strides2,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect2, expect_dims);

  const int32_t begin_indices3[] = {-1};
  const int32_t strides3[] = {-2};
  const float expect3[2] = {4, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices3, end_indices2, strides3,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect3, expect_dims);

  const int32_t begin_indices4[] = {-1};
  const int32_t strides4[] = {-2};
  const float expect4[2] = {4, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices4, end_indices2, strides4,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect4, expect_dims);

  const float expect5[3] = {4, 3, 2};
  const int32_t expect_dims5[output_dim_size] = {3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices2, strides2,
                   indices_dims, indices_dim_size,
                   1, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect5, expect_dims5);

  const float expect6[3] = {3, 2, 1};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices2, strides2,
                   indices_dims, indices_dim_size,
                   0, 1, 0, 0, 0, output_dim_size,
                   output, output_dims, expect6, expect_dims5);

  const float expect7[4] = {4, 3, 2, 1};
  const int32_t expect_dims7[output_dim_size] = {4};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices2, strides2,
                   indices_dims, indices_dim_size,
                   1, 1, 0, 0, 0, output_dim_size,
                   output, output_dims, expect7, expect_dims7);

  const int32_t begin_indices8[] = {2};
  const int32_t end_indices8[] = {4};
  const int32_t strides8[] = {2};
  const float expect8[2] = {1, 3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices8, end_indices8, strides8,
                   indices_dims, indices_dim_size,
                   1, 1, 0, 0, 0, output_dim_size,
                   output, output_dims, expect8, expect_dims);

  const int32_t output_dim_size9 = 0;
  int32_t output_dims9[] = {1};
  const float expect9[] = {3};
  const int32_t *expect_dims9 = NULL;
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices8, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 1, output_dim_size9,
                   output, output_dims9, expect9, expect_dims9);
}

void TestStridedSliceRank2() {
  const float input[] = {1, 2, 3, 4, 5, 6};
  const int32_t begin_indices[] = {0, 0};
  const int32_t end_indices[] = {2, 3};
  const int32_t strides[] = {1, 1};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {2};
  const int32_t input_dim_size = 2;
  const int32_t input_dims[input_dim_size] = {2, 3};

  float output[6] = {0};
  const int32_t output_dim_size = 2;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[6] = {1, 2, 3, 4, 5, 6};
  const int32_t expect_dims[output_dim_size] = {2, 3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const int32_t begin_indices1[] = {0};
  const int32_t end_indices1[] = {2};
  const int32_t strides1[] = {1};
  const int32_t indices_dims1[indices_dim_size] = {1};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices1, end_indices1, strides1,
                   indices_dims1, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const int32_t begin_indices2[] = {1, 1};
  const float expect2[2] = {5, 6};
  const int32_t expect_dims2[output_dim_size] = {1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices2, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect2, expect_dims2);

  const int32_t strides3[] = {1, 2};
  const float expect3[4] = {1, 3, 4, 6};
  const int32_t expect_dims3[output_dim_size] = {2, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides3,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect3, expect_dims3);

  const int32_t begin_indices4[] = {1, 2};
  const int32_t end_indices4[] = {0, 0};
  const int32_t strides4[] = {-1, -1};
  const float expect4[2] = {6, 5};
  const int32_t expect_dims4[output_dim_size] = {1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices4, end_indices4, strides4,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect4, expect_dims4);

  const float expect5[6] = {6, 5, 4, 3, 2, 1};
  const int32_t expect_dims5[output_dim_size] = {2, 3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices4, end_indices4, strides4,
                   indices_dims, indices_dim_size,
                   3, 3, 0, 0, 0, output_dim_size,
                   output, output_dims, expect5, expect_dims5);

  const int32_t begin_indices6[] = {1, 0};
  const int32_t end_indices6[] = {2, 3};
  const int32_t strides6[] = {1, 1};
  const float expect6[3] = {4, 5, 6};
  const int32_t output_dim_size6 = 1;
  const int32_t expect_dims6[output_dim_size6] = {3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices6, end_indices6, strides6,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 1, output_dim_size6,
                   output, output_dims, expect6, expect_dims6);

  const int32_t begin_indices7[] = {1, 2};
  const float expect7[1] = {6};
  const int32_t output_dim_size7 = 0;
  const int32_t *expect_dims7 = NULL;
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices7, end_indices6, strides6,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 3, output_dim_size7,
                   output, output_dims, expect7, expect_dims7);
}

void TestStridedSliceRank3() {
  const float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int32_t begin_indices[] = {0, 0, 0};
  const int32_t end_indices[] = {2, 3, 2};
  const int32_t strides[] = {1, 2, 1};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {3};
  const int32_t input_dim_size = 3;
  const int32_t input_dims[input_dim_size] = {2, 3, 2};

  float output[8] = {0};
  const int32_t output_dim_size = 3;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[8] = {1, 2, 5, 6, 7, 8, 11, 12};
  const int32_t expect_dims[output_dim_size] = {2, 2, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const float input1[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
  const int32_t begin_indices1[] = {1, 0, 0};
  const int32_t end_indices1[] = {2, 1, 3};
  const int32_t strides1[] = {1, 1, 1};
  const int32_t input_dims1[input_dim_size] = {3, 2, 3};
  const float expect1[3] = {3, 3, 3};
  const int32_t expect_dims1[output_dim_size] = {1, 1, 3};
  TestStridedSlice(input1, input_dims1, input_dim_size,
                   begin_indices1, end_indices1, strides1,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect1, expect_dims1);

  const int32_t begin_indices2[] = {0, 0, 0};
  const int32_t end_indices2[] = {2, 2, 2};
  const int32_t strides2[] = {1, 2, 1};
  const float expect2[4] = {1, 1, 3, 3};
  const int32_t expect_dims2[output_dim_size] = {2, 1, 2};
  TestStridedSlice(input1, input_dims1, input_dim_size,
                   begin_indices2, end_indices2, strides2,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect2, expect_dims2);
}

void TestStridedSliceRank4() {
  const float input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int32_t begin_indices[] = {1, 0, 1, 0};
  const int32_t end_indices[] = {2, 2, 2, 2};
  const int32_t strides[] = {1, 1, 1, 1};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {4};
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {2, 2, 2, 3};

  float output[8] = {0};
  const int32_t output_dim_size = 4;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[8] = {15, 16, 21, 22};
  const int32_t expect_dims[output_dim_size] = {1, 2, 1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect, expect_dims);

  const float expect1[8] = {3, 4, 9, 10, 15, 16, 21, 22};
  const int32_t expect_dims1[output_dim_size] = {2, 2, 1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   3, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect1, expect_dims1);

  const float expect2[8] = {15, 16, 17, 21, 22, 23};
  const int32_t expect_dims2[output_dim_size] = {1, 2, 1, 3};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 8, 0, 0, 0, output_dim_size,
                   output, output_dims, expect2, expect_dims2);

  const float expect3[8] = {15, 21};
  const int32_t output_dim_size3 = 3;
  const int32_t expect_dims3[output_dim_size3] = {1, 2, 1};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 8, 0, 0, 8, output_dim_size3,
                   output, output_dims, expect3, expect_dims3);

  const float expect4[8] = {15};
  const int32_t output_dim_size4 = 0;
  const int32_t *expect_dims4 = NULL;
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices, end_indices, strides,
                   indices_dims, indices_dim_size,
                   0, 8, 0, 0, 15, output_dim_size4,
                   output, output_dims, expect4, expect_dims4);

  const int32_t begin_indices5[] = {-1, 2, 1, 3};
  const int32_t end_indices5[] = {0, 0, 0, 0};
  const int32_t strides5[] = {-1, -1, -1, -1};

  const float expect5[2] = {23, 22};
  const int32_t expect_dims5[output_dim_size] = {1, 1, 1, 2};
  TestStridedSlice(input, input_dims, input_dim_size,
                   begin_indices5, end_indices5, strides5,
                   indices_dims, indices_dim_size,
                   0, 0, 0, 0, 0, output_dim_size,
                   output, output_dims, expect5, expect_dims5);
}

void TestSlice() {
  const float input[] = {1, 2, 3, 4, 5, 6};
  const int32_t begin_indices[] = {0, 0};
  const int32_t indice_sizes[] = {2, 3};
  const int32_t indices_dim_size = 1;
  const int32_t indices_dims[indices_dim_size] = {2};
  const int32_t input_dim_size = 2;
  const int32_t input_dims[input_dim_size] = {2, 3};

  float output[6] = {0};
  const int32_t output_dim_size = 2;
  int32_t output_dims[output_dim_size] = {0};
  const float expect[6] = {1, 2, 3, 4, 5, 6};
  const int32_t expect_dims[output_dim_size] = {2, 3};
  TestSlice(input, input_dims, input_dim_size,
            begin_indices, indice_sizes,
            indices_dims, indices_dim_size,
            output, output_dims, output_dim_size,
            expect, expect_dims);

  const int32_t begin_indices1[] = {1, 0};
  const int32_t indice_sizes1[] = {1, 2};
  const float expect1[2] = {4, 5};
  const int32_t expect_dims1[output_dim_size] = {1, 2};
  TestSlice(input, input_dims, input_dim_size,
            begin_indices1, indice_sizes1,
            indices_dims, indices_dim_size,
            output, output_dims, output_dim_size,
            expect1, expect_dims1);

  const int32_t begin_indices2[] = {0, 1};
  const int32_t indice_sizes2[] = {2, -1};
  const float expect2[4] = {2, 3, 5, 6};
  const int32_t expect_dims2[output_dim_size] = {2, 2};
  TestSlice(input, input_dims, input_dim_size,
            begin_indices2, indice_sizes2,
            indices_dims, indices_dim_size,
            output, output_dims, output_dim_size,
            expect2, expect_dims2);
}

}  // namespace


TEST_F(StridedSliceOpTest, TestStridedSliceByFirstAxis) {
  TestStridedSliceByFirstAxis();
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank1) {
  TestStridedSliceRank1();}

TEST_F(StridedSliceOpTest, TestStridedSliceRank2) {
  TestStridedSliceRank2();
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank3) {
  TestStridedSliceRank3();
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank4) {
  TestStridedSliceRank4();
}

TEST_F(StridedSliceOpTest, TestSlice) {
  TestSlice();
}

}  // namespace test
}  // namespace ops
}  // namespace micro
