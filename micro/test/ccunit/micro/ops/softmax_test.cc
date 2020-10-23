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
#include "micro/ops/softmax.h"
#include "micro/ops/gtest_utils.h"
#include "micro/ops/nhwc/cmsis_nn/arm_softmax_int8.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_quantize_utils.h"
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

  ExpectTensorNear<float>(output, output_dims, output_dim_size, expect,
                          expect_dims, output_dim_size, 1e-5);
}

}  // namespace


TEST_F(SoftmaxOpTest, CPUSimple) { Simple(); }
TEST_F(SoftmaxOpTest, CPUSimpleUseLog) { Simple(true); }

#ifdef MACE_MICRO_ENABLE_CMSIS

namespace {

void TestSoftmaxQuantInt8(const int32_t *input_dims,
                          const uint32_t input_dim_size,
                          bool use_log = false) {
  int32_t shape_size = base::GetShapeSize(input_dim_size, input_dims);
  float *input = new float[shape_size];
  FillNormalRandomInput(input, shape_size);
  float *expect_output = new float[shape_size];
  const uint32_t MAX_OUTPUT_NUM = 10;
  int32_t *expect_output_dims = new int32_t[MAX_OUTPUT_NUM];

  SoftmaxOp softmax_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, input_dim_size)
      .AddArg("use_log", static_cast<int>(use_log))
      .AddOutput(expect_output, expect_output_dims, MAX_OUTPUT_NUM);
  softmax_op.Init(
      NULL, reinterpret_cast<framework::OpContext *>(&substitude_op), NULL);
  softmax_op.Run();
  uint32_t expect_output_dim_size = substitude_op.GetOutputShapeDimSize(0);

  int8_t *input_int8 = new int8_t[shape_size];
  int8_t *output_int8 = new int8_t[shape_size];
  float *output = new float[shape_size];
  int32_t *output_dims = new int32_t[MAX_OUTPUT_NUM];
  QuantizeInfo input_quant_info;
  AutoQuantizeInt8(input, shape_size, input_int8, &input_quant_info.scale,
               &input_quant_info.zero);
  QuantizeInfo output_quant_info = {1.0f / 255.0f, -128};

  ArmSoftmaxInt8Op softmax_op_int8;
  framework::SubstituteOp substitude_op_int8;
  substitude_op_int8
      .AddInput(input_int8, input_dims, input_dim_size, input_quant_info)
      .AddArg("use_log", static_cast<int>(use_log))
      .AddOutput(output_int8, output_dims, MAX_OUTPUT_NUM, output_quant_info);
  softmax_op_int8.Init(
      NULL, reinterpret_cast<framework::OpContext *>(&substitude_op_int8),
      NULL);
  softmax_op_int8.Run();
  uint32_t output_dim_size = substitude_op_int8.GetOutputShapeDimSize(0);

  Dequantize(output_int8, shape_size, output_quant_info.scale,
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

TEST_F(SoftmaxOpTest, QuantInt8) {
  const int32_t input_dims0[2] = {5, 10};
  TestSoftmaxQuantInt8(input_dims0, 2);
  const int32_t input_dims1[2] = {50, 100};
  TestSoftmaxQuantInt8(input_dims1, 2);
  const int32_t input_dims2[2] = {1, 31};
  TestSoftmaxQuantInt8(input_dims2, 2);
}

#endif

}  // namespace test
}  // namespace ops
}  // namespace micro
