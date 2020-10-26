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
#include "micro/ops/nhwc/conv_2d_ref.h"
#include "micro/ops/nhwc/cmsis_nn/arm_conv_2d_int8.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"
#include "micro/ops/test_quantize_utils.h"

namespace micro {
namespace ops {
namespace test {

class Conv2dOpTest : public ::testing::Test {};

namespace {

void TestNHWCSimple3x3VALID() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {18.1f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3SAME() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[9] = {0};
  int32_t output_dims[4] = {0};
  float expect[9] = {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 3, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3NeqStride() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {0.1f};
  int32_t bias_dims[1] = {1};

  float output[6] = {0};
  int32_t output_dims[4] = {0};
  float expect[6] = {8.1f, 8.1f, 12.1f, 12.1f, 8.1f, 8.1f};
  int32_t expect_dims[4] = {1, 3, 2, 1};

  const int32_t strides[] = {1, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCSimple3x3WithoutBias() {
  float input[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {18.0f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestNHWCCombined3x3() {
  float input[50] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 5, 5, 2};
  float filter[36] =
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
       0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int32_t filter_dims[4] = {2, 3, 3, 2};
  float bias[2] = {0.1f, 0.2f};
  int32_t bias_dims[1] = {2};

  float output[18] = {0};
  int32_t output_dims[4] = {0};
  float expect[18] = {8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 18.1f,
                      9.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f};
  int32_t expect_dims[4] = {1, 3, 3, 2};

  const int32_t strides[] = {2, 2};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::SAME)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestFusedNHWCSimple3x3VALID(bool need_bias) {
  float input[18] =
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int32_t input_dims[4] = {1, 3, 3, 2};
  float filter[18] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int32_t filter_dims[4] = {1, 3, 3, 2};
  float bias[1] = {-0.1f};
  int32_t bias_dims[1] = {1};

  float output[1] = {0};
  int32_t output_dims[4] = {0};
  float expect[1] = {0.0f};
  int32_t expect_dims[4] = {1, 1, 1, 1};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};
  const char activation[] = "RELU";

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddRepeatArg("activation", activation, sizeof(activation))
      .AddOutput(output, output_dims, 4);
  if (need_bias) {
    substitude_op.AddInput(bias, bias_dims, 1);
  }

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

void TestConv1x1() {
  float input[150] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int32_t input_dims[4] = {1, 3, 10, 5};
  float filter[10] =
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  int32_t filter_dims[4] = {2, 1, 1, 5};
  float bias[2] = {0.1f, 0.2f};
  int32_t bias_dims[1] = {2};

  float output[60] = {0};
  int32_t output_dims[4] = {0};
  float expect[60] = {
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
      5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f};
  int32_t expect_dims[4] = {1, 3, 10, 2};

  const int32_t strides[] = {1, 1};
  const int32_t dilations[] = {1, 1};

  Conv2dRefOp conv_2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", Padding::VALID)
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);

  conv_2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);
  conv_2d_op.Run();

  ExpectTensorNear<float>(output, output_dims, 4, expect, expect_dims, 4, 1e-5);
}

}  // namespace

TEST_F(Conv2dOpTest, TestConv2dVALID) {
  TestNHWCSimple3x3VALID();
}

TEST_F(Conv2dOpTest, TestConv2dSAME) {
  TestNHWCSimple3x3SAME();
}

TEST_F(Conv2dOpTest, NotEqualStrideSimple) {
  TestNHWCSimple3x3NeqStride();
}

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestNHWCSimple3x3WithoutBias();
}

TEST_F(Conv2dOpTest, CPUStride2) {
  TestNHWCCombined3x3();
}

TEST_F(Conv2dOpTest, FusedCPUSimple) {
  TestFusedNHWCSimple3x3VALID(true);
  TestFusedNHWCSimple3x3VALID(false);
}

TEST_F(Conv2dOpTest, CPUConv1x1) {
  TestConv1x1();
}

#ifdef MACE_MICRO_ENABLE_CMSIS

namespace {

void TestConv2dQuantInt8(const int32_t batch,
                         const int32_t out_channels,
                         const int32_t in_channels,
                         const int32_t in_height,
                         const int32_t in_width,
                         const int32_t kernel_height,
                         const int32_t kernel_width,
                         enum Padding padding_type,
                         const int32_t stride_height,
                         const int32_t stride_width,
                         const int32_t dilation_height,
                         const int32_t dilation_width) {
  uint32_t input0_size = batch * in_height * in_width * in_channels;
  uint32_t input1_size =
      out_channels * kernel_height * kernel_width * in_channels;
  uint32_t max_output_size = batch * out_channels *
                             (in_height + kernel_height * dilation_height) *
                             (in_width + kernel_width * dilation_width);
  int32_t bias_size = out_channels;
  float *input0 = new float[input0_size];
  float *input1 = new float[input1_size];
  float *bias = new float[bias_size];
  FillNormalRandomInput(input0, input0_size);
  FillNormalRandomInput(input1, input1_size);
  FillNormalRandomInput(bias, bias_size);
  float *expect_output = new float[max_output_size];
  const uint32_t MAX_OUTPUT_NUM = 10;
  int32_t *expect_output_dims = new int32_t[MAX_OUTPUT_NUM];

  const int32_t input0_dims[4] = {batch, in_height, in_width, in_channels};
  const int32_t input1_dims[4] = {out_channels, kernel_height, kernel_width,
                                  in_channels};
  const int32_t bias_dims[1] = {bias_size};

  const int32_t strides[2] = {stride_height, stride_width};
  const int32_t dilations[2] = {dilation_height, dilation_width};

  Conv2dRefOp conv2d_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input0, input0_dims, 4)
      .AddInput(input1, input1_dims, 4)
      .AddInput(bias, bias_dims, 1)
      .AddArg("padding", padding_type)
      .AddRepeatArg("strides", strides, 2)
      .AddRepeatArg("dilations", dilations, 2)
      .AddOutput(expect_output, expect_output_dims, MAX_OUTPUT_NUM);
  conv2d_op.Init(NULL, reinterpret_cast<framework::OpContext *>(&substitude_op),
                 NULL);
  conv2d_op.Run();
  uint32_t expect_output_dim_size = substitude_op.GetOutputShapeDimSize(0);
  uint32_t exepct_output_size =
      base::GetShapeSize(expect_output_dim_size, expect_output_dims);

  int8_t *input0_int8 = new int8_t[input0_size];
  int8_t *input1_int8 = new int8_t[input1_size];
  int32_t *bias_int32 = new int32_t[bias_size];
  int8_t *output_int8 = new int8_t[max_output_size];
  float *output = new float[max_output_size];
  int32_t *output_dims = new int32_t[MAX_OUTPUT_NUM];
  QuantizeInfo input_quant_info0;
  QuantizeInfo input_quant_info1;
  AutoQuantizeInt8(input0, input0_size, input0_int8, &input_quant_info0.scale,
                   &input_quant_info0.zero);
  AutoQuantizeInt8Symmetric(input1, input1_size, input1_int8,
                            &input_quant_info1.scale);
  QuantizeInfo output_quant_info = {0.0f, 0};
  AdjustRangeInt8(expect_output, exepct_output_size, &output_quant_info.scale,
                  &output_quant_info.zero);
  float bias_scale = input_quant_info0.scale * input_quant_info1.scale;
  QuantizeWithScaleAndZeropoint(bias, bias_size, bias_scale, 0, bias_int32);

  ArmConv2dInt8Op conv2d_op_int8;
  framework::SubstituteOp substitude_op_int8;
  substitude_op_int8.AddInput(input0_int8, input0_dims, 4, input_quant_info0)
      .AddInput(input1_int8, input1_dims, 4, input_quant_info1)
      .AddInput(bias_int32, bias_dims, 1)
      .AddArg("padding", padding_type)
      .AddRepeatArg("strides", strides, 2)
      .AddRepeatArg("dilations", dilations, 2)
      .AddOutput(output_int8, output_dims, MAX_OUTPUT_NUM, output_quant_info);
  conv2d_op_int8.Init(
      NULL, reinterpret_cast<framework::OpContext *>(&substitude_op_int8),
      NULL);
  conv2d_op_int8.Run();
  uint32_t output_dim_size = substitude_op_int8.GetOutputShapeDimSize(0);

  uint32_t output_size = base::GetShapeSize(output_dim_size, output_dims);
  Dequantize(output_int8, output_size, output_quant_info.scale,
             output_quant_info.zero, output);

  ExpectTensorSimilar(expect_output, expect_output_dims, expect_output_dim_size,
                      output, output_dims, output_dim_size, 0.1);

  delete[] input0;
  delete[] input1;
  delete[] bias;
  delete[] expect_output;
  delete[] expect_output_dims;
  delete[] input0_int8;
  delete[] input1_int8;
  delete[] bias_int32;
  delete[] output_int8;
  delete[] output;
  delete[] output_dims;
}

}  // namespace

TEST_F(Conv2dOpTest, QuantInt8) {
  TestConv2dQuantInt8(1, 128, 64, 32, 32, 3, 3, VALID, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 64, 32, 32, 3, 3, SAME, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 64, 32, 32, 3, 3, FULL, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 64, 32, 54, 3, 3, FULL, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 512, 14, 13, 3, 3, SAME, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 64, 14, 13, 5, 5, SAME, 2, 2, 1, 1);
  TestConv2dQuantInt8(1, 128, 257, 28, 28, 3, 3, SAME, 1, 1, 1, 1);
  TestConv2dQuantInt8(1, 1, 128, 56, 56, 3, 3, SAME, 2, 2, 1, 1);
  TestConv2dQuantInt8(1, 2, 1, 1000, 1000, 4, 3, FULL, 2, 1, 1, 1);
  TestConv2dQuantInt8(1, 128, 1, 1000, 1000, 4, 3, FULL, 2, 3, 1, 1);

  // dilations is unsupported
  // TestConv2dQuantInt8(1, 128, 64, 32, 32, 3, 3, SAME, 1, 1, 2, 2);
  // TestConv2dQuantInt8(1, 128, 64, 32, 32, 3, 3, SAME, 1, 1, 2, 1);

  // batch must be 1
  // TestConv2dQuantInt8(2, 128, 64, 32, 32, 3, 3, SAME, 1, 1, 1, 1);
  // TestConv2dQuantInt8(4, 128, 64, 32, 32, 3, 3, SAME, 1, 1, 1, 1);
}

#endif

}  // namespace test
}  // namespace ops
}  // namespace micro
