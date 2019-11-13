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
#include "micro/ops/eltwise.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class EltwiseOpTest : public ::testing::Test {};

namespace {
template<typename T, typename DstType>
void SimpleScalarScalar(eltwise::Type type, T input_value,
                        float x, const DstType expect_value) {
  T input[1] = {input_value};
  int32_t input_dims[1] = {1};

  T output[1] = {0};
  int32_t output_dims[1] = {0};
  DstType expect[1] = {expect_value};
  int32_t expect_dims[1] = {1};

  EltwiseOp<T> eltwise_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 1)
      .AddArg("type", static_cast<int>(type))
      .AddArg("scalar_input", x)
      .AddOutput(output, output_dims, 1);

  eltwise_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  eltwise_op.Run();

  ExpectTensorNear<T>(output, output_dims, 1, expect, expect_dims, 1, 1e-5);
}

template<typename T, typename DstType>
void SimpleTensorScalar(eltwise::Type type, const T *input,
                        const int32_t *input_dims, const int32_t input_dim_size,
                        float x, const int32_t output_dim_size,
                        DstType *output, int32_t *output_dims,
                        const DstType *expect, const int32_t *expect_dims) {
  EltwiseOp<T> eltwise_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddArg("type", static_cast<int>(type))
      .AddArg("scalar_input", x)
      .AddOutput(output, output_dims, output_dim_size);

  eltwise_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  eltwise_op.Run();

  ExpectTensorNear<T>(output, output_dims, output_dim_size,
                      expect, expect_dims, output_dim_size, 1e-5);
}

template<typename T, typename DstType>
void SimpleTensorScalarForSpecial(eltwise::Type type, const T *input,
                                  float x, const DstType *expect) {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {1, 1, 2, 3};
  const int32_t output_dim_size = 4;
  DstType output[6] = {0};
  int32_t output_dims[output_dim_size] = {0};
  const int32_t expect_dims[output_dim_size] = {1, 1, 2, 3};
  SimpleTensorScalar(type, input, input_dims,
                     input_dim_size, x, output_dim_size,
                     output, output_dims,
                     expect, expect_dims);
}

void SimpleTensorScalar1() {
  const int32_t input_dim_size = 4;
  const int32_t input_dims[input_dim_size] = {1, 1, 1, 1};
  const float input[] = {1};
  const int32_t output_dim_size = 4;
  float output[1] = {0};
  int32_t output_dims[output_dim_size] = {0};
  const float expect[1] = {2};
  const int32_t expect_dims[output_dim_size] = {1, 1, 1, 1};
  SimpleTensorScalar(eltwise::SUM, input, input_dims,
                     input_dim_size, 1, output_dim_size,
                     output, output_dims,
                     expect, expect_dims);
}

template<typename T, typename DstType>
void SimpleTensorEltwise(eltwise::Type type, const T *input0,
                         const int32_t *input0_dims,
                         const int32_t input0_dim_size,
                         const T *input1, const int32_t *input1_dims,
                         const int32_t input1_dim_size,
                         DstType *output, int32_t *output_dims,
                         const int32_t output_dim_size,
                         const DstType *expect, const int32_t *expect_dims,
                         const float *coeff = NULL,
                         const uint32_t coeff_len = 0) {
  EltwiseOp<T> eltwise_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input0, input0_dims, input0_dim_size)
      .AddArg("type", static_cast<int>(type))
      .AddOutput(output, output_dims, output_dim_size);
  if (input1 != NULL && input1_dims != NULL && input1_dim_size > 0) {
    substitude_op.AddInput(input1, input1_dims, input1_dim_size);
  }

  if (coeff != NULL && coeff_len > 0) {
    substitude_op.AddRepeatArg("coeff", coeff, coeff_len);
  }

  eltwise_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  eltwise_op.Run();

  ExpectTensorNear<T>(output, output_dims, output_dim_size,
                      expect, expect_dims, output_dim_size, 1e-5);
}

template<typename T, typename DstType>
void SimpleTensorEltwise(eltwise::Type type, const T *input0,
                         const int32_t *input0_dims, const T *input1,
                         const int32_t *input1_dims, DstType *output,
                         const DstType *expect, const int32_t *expect_dims,
                         const float *coeff = NULL,
                         const uint32_t coeff_len = 0) {
  int32_t output_dims[4] = {0};
  SimpleTensorEltwise(type, input0, input0_dims, 4, input1, input1_dims, 4,
                      output, output_dims, 4, expect, expect_dims, coeff,
                      coeff_len);
}
}  // namespace

TEST_F(EltwiseOpTest, SimpleScalarScalar) {
  SimpleScalarScalar<float, float>(eltwise::SUM, 1, 2, 3);
  SimpleScalarScalar<float, float>(eltwise::SUB, 1, 2, -1);
  SimpleScalarScalar<float, float>(eltwise::PROD, 1, 2, 2);
  SimpleScalarScalar<float, float>(eltwise::DIV, 1, 2, 0.5);
  SimpleScalarScalar<float, float>(eltwise::FLOOR_DIV, 1, 2, 0);
  SimpleScalarScalar<float, float>(eltwise::FLOOR_DIV, 1, -2, -1);
  SimpleScalarScalar<float, float>(eltwise::MIN, 1, 2, 1);
  SimpleScalarScalar<float, float>(eltwise::MAX, 1, 2, 2);
  SimpleScalarScalar<float, float>(eltwise::NEG, 1, 2, -1);
  SimpleScalarScalar<float, float>(eltwise::ABS, -1, 3, 1);
  SimpleScalarScalar<float, float>(eltwise::SIGN, -2, 3, -1);
  SimpleScalarScalar<int32_t, int32_t>(eltwise::EQUAL, 1, 3, 0);
  SimpleScalarScalar<int32_t, int32_t>(eltwise::EQUAL, 3, 3, 1);
}

TEST_F(EltwiseOpTest, CPUSimpleTensorScalar) {
  SimpleTensorScalar1();
  const float input[] = {1, 2, 3, 4, 5, 6};
  const float expect2[] = {0, 1, 2, 3, 4, 5};
  SimpleTensorScalarForSpecial<float, float>(eltwise::SUB, input, 1, expect2);

  const float expect3[] = {2, 4, 6, 8, 10, 12};
  SimpleTensorScalarForSpecial<float, float>(eltwise::PROD, input, 2, expect3);

  const float expect4[] = {1, 1, 1, 1, 1, 1};
  SimpleTensorScalarForSpecial<float, float>(eltwise::MIN, input, 1, expect4);

  const float expect5[] = {3, 3, 3, 4, 5, 6};
  SimpleTensorScalarForSpecial<float, float>(eltwise::MAX, input, 3, expect5);

  const float expect6[] = {-1, -2, -3, -4, -5, -6};
  SimpleTensorScalarForSpecial<float, float>(eltwise::NEG, input, 3, expect6);

  const float expect7[] = {0, 1, 4, 9, 16, 25};
  SimpleTensorScalarForSpecial<float, float>(
      eltwise::SQR_DIFF, input, 1, expect7);

  const int32_t input_i[] = {1, 2, 3, 4, 5, 6};
  const int32_t expect8[] = {0, 0, 1, 0, 0, 0};
  SimpleTensorScalarForSpecial<int32_t, int32_t>(
      eltwise::EQUAL, input_i, 3, expect8);

  const float input9[] = {2, 4, 6, 8, 10, 12};
  const float expect9[] = {1, 2, 3, 4, 5, 6};
  SimpleTensorScalarForSpecial<float, float>(eltwise::DIV, input9, 2, expect9);

  const float expect10[] = {0, 1, 2, 2, 3, 4};
  SimpleTensorScalarForSpecial<float, float>(
      eltwise::FLOOR_DIV, input9, 3, expect10);

  const float expect11[] = {-1, -2, -2, -3, -4, -4};
  SimpleTensorScalarForSpecial<float, float>(
      eltwise::FLOOR_DIV, input9, -3, expect11);

  const float input12[] = {-1, -2, -3, -4, -5, -6};
  const float expect12[] = {1, 2, 3, 4, 5, 6};
  SimpleTensorScalarForSpecial<float, float>(
      eltwise::ABS, input12, 3, expect12);

  const float input13[] = {1, 2, -3, 0, -5, -6};
  const float expect13[] = {1, 1, -1, 0, -1, -1};
  SimpleTensorScalarForSpecial<float, float>(
      eltwise::SIGN, input13, 3, expect13);
}

TEST_F(EltwiseOpTest, CPUSimpleTensorVector) {
  const int32_t dims1123[] = {1, 1, 2, 3};
  const int32_t dims1113[] = {1, 1, 1, 3};
  const int32_t dims1215[] = {1, 2, 1, 5};
  const int32_t dims1115[] = {1, 1, 1, 5};
  const int32_t dims1213[] = {1, 2, 1, 3};
  const int32_t dims3[] = {3};
  const int32_t dims5[] = {5};

  float output6[6] = {0};
  float output10[10] = {0};
  int32_t output6_i[6] = {0};

  int32_t output_dims4[4] = {0};

  const float input0_0[] = {1, 2, 3, 4, 5, 6};
  const float input1_0[] = {1, 2, 3};
  const float expect_0[] = {2, 4, 6, 5, 7, 9};
  SimpleTensorEltwise(eltwise::SUM, input0_0, dims1123, input1_0,
                      dims1113, output6, expect_0, dims1123);

  const float input0_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const float input1_1[] = {1, 2, 3, 4, 5};
  const float expect_1[] = {0, 0, 0, 0, 0, 5, 5, 5, 5, 5};
  SimpleTensorEltwise(eltwise::SUB, input0_1, dims1215, input1_1,
                      dims1115, output10, expect_1, dims1215);

  const float expect_2[] = {0, 0, 0, 0, 0, -5, -5, -5, -5, -5};
  SimpleTensorEltwise(eltwise::SUB, input1_1, dims1115, input0_1,
                      dims1215, output10, expect_2, dims1215);

  const float expect_3[] = {1, 4, 9, 4, 10, 18};
  SimpleTensorEltwise(eltwise::PROD, input1_0, dims1113, input0_0,
                      dims1213, output6, expect_3, dims1213);

  const float input1_4[] = {1, 1, 1, 1, 5};
  const float expect_4[] = {1, 2, 3, 4, 1, 6, 7, 8, 9, 2};
  SimpleTensorEltwise(eltwise::DIV, input0_1, dims1215, input1_4,
                      dims1115, output10, expect_4, dims1215);

  const float input0_5[] = {1, 1, 1, 2, 4};
  const float input1_5[] = {1, 1, 1, 2, 2, 1, 1, 1, 1, 1};
  const float expect_5[] = {1, 1, 1, 1, 2, 1, 1, 1, 2, 4};
  SimpleTensorEltwise(eltwise::DIV, input0_5, dims1115, input1_5,
                      dims1215, output10, expect_5, dims1215);

  const float input1_6[] = {2, 2, 2, 2, 3};
  const float expect_6[] = {0, 1, 1, 2, 1, 3, 3, 4, 4, 3};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_1, dims1215, input1_6,
                      dims1115, output10, expect_6, dims1215);

  const float input1_7[] = {-2, -2, -2, -2, -3};
  const float expect_7[] = {-1, -1, -2, -2, -2, -3, -4, -4, -5, -4};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_1, dims1215, input1_7,
                      dims1115, output10, expect_7, dims1215);

  const float input1_8[] = {2, 2, 2, 3, 3, 2, 2, 2, 2, 2};
  const float expect_8[] = {0, 0, 0, 0, 1, 0, 0, 0, 1, 2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_5, dims1115, input1_8,
                      dims1215, output10, expect_8, dims1215);

  const float input1_9[] = {-2, -2, -2, -3, -3, -2, -2, -2, -2, -2};
  const float expect_9[] = {-1, -1, -1, -1, -2, -1, -1, -1, -1, -2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_5, dims1115, input1_9,
                      dims1215, output10, expect_9, dims1215);

  const float expect_10[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
  SimpleTensorEltwise(eltwise::MIN, input1_1, dims1115, input0_1,
                      dims1215, output10, expect_10, dims1215);

  const float expect_11[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimpleTensorEltwise(eltwise::MAX, input0_1, dims1215, input1_1,
                      dims1115, output10, expect_11, dims1215);

  const float expect_12[] = {0, 0, 0, 0, 0, 25, 25, 25, 25, 25};
  SimpleTensorEltwise(eltwise::SQR_DIFF, input1_1, dims1115, input0_1,
                      dims1215, output10, expect_12, dims1215);

  const int32_t input0_13[] = {1, 2, 3, 4, 5, 6};
  const int32_t input1_13[] = {1, 2, 3};
  const int32_t expect_13[] = {1, 1, 1, 0, 0, 0};
  SimpleTensorEltwise(eltwise::EQUAL, input0_13, dims1123, input1_13,
                      dims1113, output6_i, expect_13, dims1123);

  const float expect_14[] = {2, 4, 6, 5, 7, 9};
  SimpleTensorEltwise(eltwise::SUM, input0_0, dims1123,
                      4, input1_0, dims3, 1, output6,
                      output_dims4, 4, expect_14, dims1123);

  const float expect_15[] = {0, 0, 0, 0, 0, 5, 5, 5, 5, 5};
  SimpleTensorEltwise(eltwise::SUB, input0_1, dims1215,
                      4, input1_1, dims5, 1, output10,
                      output_dims4, 4, expect_15, dims1215);

  const float expect_16[] = {0, 0, 0, 0, 0, -5, -5, -5, -5, -5};
  SimpleTensorEltwise(eltwise::SUB, input1_1, dims5,
                      1, input0_1, dims1215, 4, output10,
                      output_dims4, 4, expect_16, dims1215);

  const float expect_17[] = {1, 4, 9, 4, 10, 18};
  SimpleTensorEltwise(eltwise::PROD, input1_0, dims3,
                      1, input0_0, dims1213, 4, output6,
                      output_dims4, 4, expect_17, dims1213);

  const float expect_18[] = {1, 2, 3, 4, 1, 6, 7, 8, 9, 2};
  SimpleTensorEltwise(eltwise::DIV, input0_1, dims1215,
                      4, input1_4, dims5, 1, output10,
                      output_dims4, 4, expect_18, dims1215);

  const float expect_19[] = {1, 1, 1, 1, 2, 1, 1, 1, 2, 4};
  SimpleTensorEltwise(eltwise::DIV, input0_5, dims5,
                      1, input1_5, dims1215, 4, output10,
                      output_dims4, 4, expect_19, dims1215);

  const float expect_20[] = {0, 1, 1, 2, 1, 3, 3, 4, 4, 3};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_1, dims1215,
                      4, input1_6, dims5, 1, output10,
                      output_dims4, 4, expect_20, dims1215);

  const float expect_21[] = {-1, -1, -2, -2, -2, -3, -4, -4, -5, -4};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_1, dims1215,
                      4, input1_7, dims5, 1, output10, output_dims4,
                      4, expect_21, dims1215);

  const float expect_22[] = {0, 0, 0, 0, 1, 0, 0, 0, 1, 2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_5, dims5, 1, input1_8,
                      dims1215, 4, output10, output_dims4,
                      4, expect_22, dims1215);

  const float expect_23[] = {-1, -1, -1, -1, -2, -1, -1, -1, -1, -2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_5, dims5, 1, input1_9,
                      dims1215, 4, output10, output_dims4,
                      4, expect_23, dims1215);

  const float expect_24[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
  SimpleTensorEltwise(eltwise::MIN, input1_1, dims5, 1, input0_1,
                      dims1215, 4, output10, output_dims4,
                      4, expect_24, dims1215);

  const float expect_25[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimpleTensorEltwise(eltwise::MAX, input0_1, dims1215, 4, input1_1,
                      dims5, 1, output10, output_dims4, 4,
                      expect_25, dims1215);

  const float expect_26[] = {0, 0, 0, 0, 0, 25, 25, 25, 25, 25};
  SimpleTensorEltwise(eltwise::SQR_DIFF, input1_1, dims5, 1, input0_1,
                      dims1215, 4, output10, output_dims4, 4,
                      expect_26, dims1215);

  const int32_t expect_27[] = {1, 1, 1, 0, 0, 0};
  SimpleTensorEltwise(eltwise::EQUAL, input0_13, dims1123, 4, input1_13,
                      dims3, 1, output6_i, output_dims4, 4,
                      expect_27, dims1123);
}

TEST_F(EltwiseOpTest, CPUSimpleTensorTensor) {
  const int32_t dims1123[] = {1, 1, 2, 3};
  const int32_t dims1215[] = {1, 2, 1, 5};
  const int32_t dims1115[] = {1, 1, 1, 5};
  const int32_t dims1213[] = {1, 2, 1, 3};

  float output6[6] = {0};
  float output10[10] = {0};
  int32_t output6_i[6] = {0};

  int32_t output_dims4[4] = {0};

  const float input0_0[] = {1, 2, 3, 4, 5, 6};
  const float expect_0[] = {2, 4, 6, 8, 10, 12};
  SimpleTensorEltwise(eltwise::SUM, input0_0, dims1123, input0_0,
                      dims1123, output6, expect_0, dims1123);

  const float expect_1[] = {0.2, 0.4, 0.6, 0.8, 1, 1.2};
  const float coeff_1[] = {0.1, 0.1};
  SimpleTensorEltwise(eltwise::SUM, input0_0, dims1123, input0_0,
                      dims1123, output6, expect_1, dims1123, coeff_1,
                      sizeof(coeff_1)/ sizeof(float));

  const float input0_2[] = {1, 2, 3, 4, 5};
  const float expect_2[] = {0, 0, 0, 0, 0};
  SimpleTensorEltwise(eltwise::SUB, input0_2, dims1115, input0_2,
                      dims1115, output6, expect_2, dims1115);

  const float expect_3[] = {1, 4, 9, 16, 25, 36};
  SimpleTensorEltwise(eltwise::PROD, input0_0, dims1213, input0_0,
                      dims1213, output6, expect_3, dims1213);

  const float expect_4[] = {1, 1, 1, 1, 1, 1};
  SimpleTensorEltwise(eltwise::DIV, input0_0, dims1213, input0_0,
                      dims1213, output6, expect_4, dims1213);

  const float input0_5[] = {2, 3, 4, 5, 6, 7};
  const float expect_5[] = {2, 1, 1, 1, 1, 1};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_5, dims1213, input0_0,
                      dims1213, output6, expect_5, dims1213);

  const float input0_6[] = {-2, -3, -4, -5, -6, -7};
  const float expect_6[] = {-2, -2, -2, -2, -2, -2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_6, dims1213, input0_0,
                      dims1213, output6, expect_6, dims1213);

  const float input0_7[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
  const float input1_7[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const float expect_7[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
  SimpleTensorEltwise(eltwise::MIN, input0_7, dims1215, input1_7,
                      dims1215, output10, expect_7, dims1215);

  const float expect_8[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimpleTensorEltwise(eltwise::MAX, input1_7, dims1215, input0_7,
                      dims1215, output10, expect_8, dims1215);

  const float expect_9[] = {0, 0, 0, 0, 0, 25, 25, 25, 25, 25};
  SimpleTensorEltwise(eltwise::SQR_DIFF, input0_7, dims1215, input1_7,
                      dims1215, output10, expect_9, dims1215);

  const int input0_10[] = {1, 2, 3, 4, 5, 6};
  const int expect_10[] = {1, 1, 1, 1, 1, 1};
  SimpleTensorEltwise(eltwise::EQUAL, input0_10, dims1123, input0_10,
                      dims1123, output6_i, expect_10, dims1123);

  const float expect_11[] = {2, 2, 3, 3, 3, 2, 2, 3, 3, 3};
  const float coeff_11[] = {2.0f, 3.0f};
  SimpleTensorEltwise<float, float>(
      eltwise::CLIP, input0_7, dims1215,
      4, NULL, NULL, 0, output10, output_dims4, 4, expect_11, dims1215,
      coeff_11, sizeof(coeff_11) / sizeof(float));
}

TEST_F(EltwiseOpTest, TensorGeneralBroadcastCPU) {
  const int32_t dims1123[] = {1, 1, 2, 3};
  const int32_t dims1121[] = {1, 1, 2, 1};

  float output[10] = {0};
  const float input0_0[] = {1, 2, 3, 4, 5, 6};
  const float input1_0[] = {1, 2};
  const float expect_0[] = {2, 3, 4, 6, 7, 8};
  SimpleTensorEltwise(eltwise::SUM, input0_0, dims1123, input1_0,
                      dims1121, output, expect_0, dims1123);

  const float expect_1[] = {0, 1, 2, 2, 3, 4};
  SimpleTensorEltwise(eltwise::SUB, input0_0, dims1123, input1_0,
                      dims1121, output, expect_1, dims1123);

  const float expect_2[] = {1, 2, 3, 8, 10, 12};
  SimpleTensorEltwise(eltwise::PROD, input0_0, dims1123, input1_0,
                      dims1121, output, expect_2, dims1123);

  const float expect_3[] = {1, 2, 3, 2, 2.5, 3};
  SimpleTensorEltwise(eltwise::DIV, input0_0, dims1123, input1_0,
                      dims1121, output, expect_3, dims1123);

  const float input1_4[] = {2, 3};
  const float expect_4[] = {0, 1, 1, 1, 1, 2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_0, dims1123, input1_4,
                      dims1121, output, expect_4, dims1123);

  const float input1_5[] = {-2, -3};
  const float expect_5[] = {-1, -1, -2, -2, -2, -2};
  SimpleTensorEltwise(eltwise::FLOOR_DIV, input0_0, dims1123, input1_5,
                      dims1121, output, expect_5, dims1123);

  const float expect_6[] = {1, 1, 1, 2, 2, 2};
  SimpleTensorEltwise(eltwise::MIN, input0_0, dims1123, input1_0,
                      dims1121, output, expect_6, dims1123);

  const float expect_7[] = {1, 2, 3, 4, 5, 6};
  SimpleTensorEltwise(eltwise::MAX, input0_0, dims1123, input1_0,
                      dims1121, output, expect_7, dims1123);

  const float expect_8[] = {0, 1, 4, 4, 9, 16};
  SimpleTensorEltwise(eltwise::SQR_DIFF, input0_0, dims1123, input1_0,
                      dims1121, output, expect_8, dims1123);

  const int32_t input0_9[] = {1, 2, 3, 4, 5, 6};
  const int32_t input1_9[] = {1, 2};
  const int32_t expect_9[] = {1, 0, 0, 0, 0, 0};
  int32_t output_9[6] = {0};
  SimpleTensorEltwise(eltwise::EQUAL, input0_9, dims1123, input1_9,
                      dims1121, output_9, expect_9, dims1123);
}

}  // namespace test
}  // namespace ops
}  // namespace micro
