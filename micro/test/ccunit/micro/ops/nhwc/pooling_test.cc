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
#include "micro/ops/nhwc/pooling_ref.h"
#include "micro/ops/nhwc/pooling_s4.h"
#include "micro/ops/nhwc/cmsis_nn/arm_pooling_int8.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_quantize_utils.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

class PoolingOpTest : public ::testing::Test {};

namespace {

void TestPoolingOpValidMax() {
  float input[32] = {
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  int32_t input_dims[4] = {1, 4, 4, 2};

  float output[8] = {0};
  int32_t output_dims[4] = {0};
  float expect[8] = {5, 21, 7, 23, 13, 29, 15, 31};
  int32_t expect_dims[4] = {1, 2, 2, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpSameMax() {
  float input[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  int32_t input_dims[4] = {1, 3, 3, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {4, 5, 7, 8};
  int32_t expect_dims[4] = {1, 2, 2, 1};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpValidDilation() {
  float input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int32_t input_dims[4] = {1, 4, 4, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {10, 11, 14, 15};
  int32_t expect_dims[4] = {1, 2, 2, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {2, 2};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::MAX)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpValidAvg() {
  float input[32] = {
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  int32_t input_dims[4] = {1, 4, 4, 2};

  float output[8] = {0};
  int32_t output_dims[4] = {0};
  float expect[8] = {2.5, 18.5, 4.5, 20.5, 10.5, 26.5, 12.5, 28.5};
  int32_t expect_dims[4] = {1, 2, 2, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddArg("pooling_type", PoolingType::AVG)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestPoolingOpSameAvg() {
  float input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int32_t input_dims[4] = {1, 2, 8, 1};

  float output[4] = {0};
  int32_t output_dims[4] = {0};
  float expect[4] = {4.5, 6.5, 8.5, 10.5};
  int32_t expect_dims[4] = {1, 1, 4, 1};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};
  const int32_t kernels[] = {2, 2};

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddArg("pooling_type", PoolingType::AVG)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  pooling_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

}  // namespace

TEST_F(PoolingOpTest, TestPoolingValidMax) {
  TestPoolingOpValidMax();
}

TEST_F(PoolingOpTest, TestPoolingSameMax) {
  TestPoolingOpSameMax();
}

TEST_F(PoolingOpTest, TestPoolingValidDilation) {
  TestPoolingOpValidDilation();
}

TEST_F(PoolingOpTest, TestPoolingOpValidAvg) {
  TestPoolingOpValidAvg();
}

TEST_F(PoolingOpTest, TestPoolingOpSameAvg) {
  TestPoolingOpSameAvg();
}

#ifdef MACE_MICRO_ENABLE_CMSIS

namespace {

void TestPoolingQuantInt8(const int32_t *input_dims,
                          const uint32_t input_dim_size,
                          const int32_t *kernels,
                          const int32_t *strides,
                          Padding padding,
                          PoolingType pooling_type) {
  int32_t input_size = base::GetShapeSize(input_dim_size, input_dims);
  int32_t max_output_size = input_dims[0] * input_dims[3] *
                            (input_dims[1] + kernels[0]) *
                            (input_dims[2] + kernels[1]);

  float *input = new float[input_size];
  FillNormalRandomInput(input, input_size);
  float *expect_output = new float[max_output_size];
  const uint32_t MAX_OUTPUT_DIM_SIZE = 100;
  int32_t *expect_output_dims = new int32_t[MAX_OUTPUT_DIM_SIZE];

  const int32_t dilations[2] = {1, 1};

  PoolingRefOp pooling_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddRepeatArg("strides", strides, 2)
      .AddRepeatArg("kernels", kernels, 2)
      .AddRepeatArg("dilations", dilations, 2)
      .AddArg("padding", padding)
      .AddArg("pooling_type", pooling_type)
      .AddOutput(expect_output, expect_output_dims, MAX_OUTPUT_DIM_SIZE);
  pooling_op.Init(
      NULL, reinterpret_cast<framework::OpContext *>(&substitude_op), NULL);
  pooling_op.Run();
  uint32_t expect_output_dim_size = substitude_op.GetOutputShapeDimSize(0);

  int8_t *input_int8 = new int8_t[input_size];
  int8_t *output_int8 = new int8_t[max_output_size];
  float *output = new float[max_output_size];
  int32_t *output_dims = new int32_t[MAX_OUTPUT_DIM_SIZE];
  QuantizeInfo input_quant_info;
  AutoQuantizeInt8(input, input_size, input_int8, &input_quant_info.scale,
               &input_quant_info.zero);
  QuantizeInfo output_quant_info = input_quant_info;

  ArmPoolingInt8Op pooling_op_int8;
  framework::SubstituteOp substitude_op_int8;
  substitude_op_int8
      .AddInput(input_int8, input_dims, input_dim_size, input_quant_info)
      .AddRepeatArg("strides", strides, 2)
      .AddRepeatArg("kernels", kernels, 2)
      .AddRepeatArg("dilations", dilations, 2)
      .AddArg("padding", padding)
      .AddArg("pooling_type", pooling_type)
      .AddOutput(output_int8, output_dims, MAX_OUTPUT_DIM_SIZE,
                 output_quant_info);
  pooling_op_int8.Init(
      NULL, reinterpret_cast<framework::OpContext *>(&substitude_op_int8),
      NULL);
  pooling_op_int8.Run();
  uint32_t output_dim_size = substitude_op_int8.GetOutputShapeDimSize(0);

  uint32_t output_size = base::GetShapeSize(output_dim_size, output_dims);
  Dequantize(output_int8, output_size, output_quant_info.scale,
             output_quant_info.zero, output);

  ExpectTensorSimilar(expect_output, expect_output_dims, expect_output_dim_size,
                      output, output_dims, output_dim_size, 0.1);

  delete[] input;
  delete[] expect_output;
  delete[] expect_output_dims;
  delete[] input_int8;
  delete[] output_int8;
  delete[] output;
  delete[] output_dims;
}

}  // namespace
TEST_F(PoolingOpTest, Quant) {
  const int32_t input_dims0[4] = {1, 7, 7, 1024};
  const int32_t kernels0[2] = {7, 7};
  const int32_t strides0[2] = {1, 1};
  TestPoolingQuantInt8(input_dims0, 4, kernels0, strides0, Padding::VALID,
                       PoolingType::AVG);
  TestPoolingQuantInt8(input_dims0, 4, kernels0, strides0, Padding::VALID,
                       PoolingType::MAX);
  TestPoolingQuantInt8(input_dims0, 4, kernels0, strides0, Padding::FULL,
                       PoolingType::AVG);
  TestPoolingQuantInt8(input_dims0, 4, kernels0, strides0, Padding::SAME,
                       PoolingType::MAX);
  const int32_t input_dims1[4] = {1, 3, 3, 2};
  const int32_t kernels1[2] = {3, 3};
  const int32_t strides1[2] = {1, 1};
  TestPoolingQuantInt8(input_dims1, 4, kernels1, strides1, Padding::SAME,
                       PoolingType::AVG);
  const int32_t input_dims2[4] = {1, 3, 3, 2};
  const int32_t kernels2[2] = {2, 3};
  const int32_t strides2[2] = {1, 2};
  TestPoolingQuantInt8(input_dims2, 4, kernels2, strides2, Padding::SAME,
                       PoolingType::MAX);
  // WARNING(ZhangZhimin): Batch inputs is unsupported
  // const int32_t input_dims3[4] = {3,15,15,128};
  // const int32_t kernels3[2] = {4, 4};
  // const int32_t strides3[2] = {4, 4};
  // TestPoolingQuantInt8(input_dims3, 4, kernels3, strides3, Padding::SAME,
  //                      PoolingType::AVG);
  // const int32_t input_dims4[4] = {3,15,15,128};
  // const int32_t kernels4[2] = {4, 4};
  // const int32_t strides4[2] = {4, 4};
  // TestPoolingQuantInt8(input_dims4, 4, kernels4, strides4, Padding::SAME,
  //                      PoolingType::MAX);
  const int32_t input_dims5[4] = {1, 31, 31, 127};
  const int32_t kernels5[2] = {2, 2};
  const int32_t strides5[2] = {3, 3};
  TestPoolingQuantInt8(input_dims5, 4, kernels5, strides5, Padding::SAME,
                       PoolingType::AVG);
  const int32_t input_dims6[4] = {1, 31, 31, 127};
  const int32_t kernels6[2] = {2, 2};
  const int32_t strides6[2] = {3, 3};
  TestPoolingQuantInt8(input_dims6, 4, kernels6, strides6, Padding::SAME,
                       PoolingType::MAX);
}

#endif


}  // namespace test
}  // namespace ops
}  // namespace micro
