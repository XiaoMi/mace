// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <fstream>

#include "mace/core/operator.h"
#include "mace/kernels/quantize.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class FullyConnectedOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const std::vector<index_t> &input_shape,
            const std::vector<float> &input_value,
            const std::vector<index_t> &weight_shape,
            const std::vector<float> &weight_value,
            const std::vector<index_t> &bias_shape,
            const std::vector<float> &bias_value,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output_value) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input_value);
  net.AddInputFromArray<D, float>("Weight", weight_shape, weight_value);
  net.AddInputFromArray<D, float>("Bias", bias_shape, bias_value);

  if (D == DeviceType::CPU) {
    OpDefBuilder("FullyConnected", "FullyConnectedTest")
        .Input("Input")
        .Input("Weight")
        .Input("Bias")
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>("OutputNCHW", NCHW, "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Weight", "WeightImage",
                            kernels::BufferType::WEIGHT_WIDTH);
    BufferToImage<D, float>(&net, "Bias", "BiasImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("FullyConnected", "FullyConnectedTest")
        .Input("InputImage")
        .Input("WeightImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>(output_shape, output_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(FullyConnectedOpTest, SimpleCPU) {
  Simple<DeviceType::CPU>({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 2, 2},
                          {1, 2, 3, 4, 5, 6, 7, 8}, {1}, {2}, {1, 1, 1, 1},
                          {206});
  Simple<DeviceType::CPU>(
      {1, 1, 2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 1, 2, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
      {2}, {2, 3}, {1, 1, 1, 2}, {387, 3853});
  Simple<DeviceType::CPU>(
      {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {5, 1, 2, 3},
      {1, 2, 3, 4,  5,  6,  10, 20, 30, 40, 50, 60, 1, 2, 3,
       4, 5, 6, 10, 20, 30, 40, 50, 60, 1,  2,  3,  4, 5, 6},
      {5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5}, {92, 912, 94, 914, 96});
}

TEST_F(FullyConnectedOpTest, SimpleCPUWithBatch) {
  Simple<DeviceType::CPU>({2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 2},
                          {1, 2, 3, 4}, {1}, {2}, {2, 1, 1, 1}, {32, 72});
}

TEST_F(FullyConnectedOpTest, SimpleOPENCL) {
  Simple<DeviceType::GPU>({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 2, 2},
                          {1, 3, 5, 7, 2, 4, 6, 8}, {1}, {2}, {1, 1, 1, 1},
                          {206});
  Simple<DeviceType::GPU>(
      {1, 1, 2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 5, 1, 2},
      {1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 10, 60, 20, 70, 30, 80, 40, 90, 50, 100},
      {2}, {2, 3}, {1, 1, 1, 2}, {387, 3853});
  Simple<DeviceType::GPU>(
      {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {5, 3, 1, 2},
      {1, 4, 2, 5,  3,  6,  10, 40, 20, 50, 30, 60, 1, 4, 2,
       5, 3, 6, 10, 40, 20, 50, 30, 60, 1,  4,  2,  5, 3, 6},
      {5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5}, {92, 912, 94, 914, 96});
}

TEST_F(FullyConnectedOpTest, SimpleGPUWithBatch) {
  Simple<DeviceType::GPU>({2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 1, 2},
                          {1, 3, 2, 4}, {1}, {2}, {2, 1, 1, 1}, {32, 72});
}

namespace {
template <typename T>
void Random(const index_t batch,
            const index_t height,
            const index_t width,
            const index_t channels,
            const index_t out_channel) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>(
      "Weight", {out_channel, channels, height, width});
  net.AddRandomInput<DeviceType::GPU, float>("Bias", {out_channel});

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);
  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("InputNCHW")
      .Input("Weight")
      .Input("Bias")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<CPU, float>("OutputNCHW", NCHW, "Output", NHWC);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::GPU, T>(&net, "Input", "InputImage",
                                    kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, T>(&net, "Weight", "WeightImage",
                                    kernels::BufferType::WEIGHT_WIDTH);
  BufferToImage<DeviceType::GPU, T>(&net, "Bias", "BiasImage",
                                    kernels::BufferType::ARGUMENT);

  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("InputImage")
      .Input("WeightImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-1,
                            1e-1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2,
                            1e-3);
  }
}
}  // namespace

TEST_F(FullyConnectedOpTest, ComplexAligned) {
  Random<float>(1, 16, 16, 32, 16);
  Random<float>(1, 7, 7, 32, 16);
  Random<float>(1, 7, 7, 512, 128);
  Random<float>(1, 1, 1, 2048, 1024);
}

TEST_F(FullyConnectedOpTest, ComplexUnAlignedWithoutBatch) {
  Random<float>(1, 13, 11, 11, 17);
  Random<float>(1, 23, 29, 23, 113);
  Random<float>(1, 14, 14, 13, 23);
}

TEST_F(FullyConnectedOpTest, ComplexMultiBatch) {
  Random<float>(11, 7, 7, 32, 16);
  Random<float>(5, 7, 7, 512, 128);
  Random<float>(3, 1, 1, 2048, 1024);
  Random<float>(7, 14, 14, 13, 23);
}

TEST_F(FullyConnectedOpTest, ComplexHalfWidthFormatAligned) {
  Random<half>(1, 2, 2, 512, 2);
  Random<half>(1, 11, 11, 32, 16);
  Random<half>(1, 16, 32, 32, 32);
  Random<half>(1, 14, 14, 13, 23);
}

namespace {
void QuantRandom(const index_t batch,
                 const index_t height,
                 const index_t width,
                 const index_t channels,
                 const index_t out_channel) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<CPU, float>(
      "Input", {batch, height, width, channels});
  net.AddRandomInput<CPU, float>(
      "Weight", {out_channel, height, width, channels});
  net.AddRandomInput<CPU, float>("Bias", {out_channel});
  net.TransformDataFormat<CPU, float>("Input", NHWC, "InputNCHW", NCHW);
  net.TransformDataFormat<CPU, float>("Weight", OHWI, "WeightOIHW", OIHW);

  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("InputNCHW")
      .Input("WeightOIHW")
      .Input("Bias")
      .Output("OutputNCHW")
      .AddIntArg("T", DT_FLOAT)
      .Finalize(net.NewOperatorDef());
  net.RunOp();
  net.TransformDataFormat<CPU, float>("OutputNCHW", NCHW, "Output", NHWC);

  OpDefBuilder("Quantize", "QuantizeWeight")
      .Input("Weight")
      .Output("QuantizedWeight")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeOutput")
      .Input("Output")
      .Output("ExpectedQuantizedOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  Tensor *q_weight = net.GetTensor("QuantizedWeight");
  Tensor *q_input = net.GetTensor("QuantizedInput");
  Tensor *bias = net.GetTensor("Bias");
  auto bias_data = bias->data<float>();
  std::vector<int32_t> q_bias(bias->size());
  kernels::QuantizeWithScaleAndZeropoint(
      bias_data, bias->size(), q_input->scale() * q_weight->scale(), 0,
      q_bias.data());
  net.AddInputFromArray<DeviceType::CPU, int32_t>("QuantizedBias",
                                                  {out_channel}, q_bias);

  OpDefBuilder("FullyConnected", "QuantizeFullyConnectedTest")
      .Input("QuantizedInput")
      .Input("QuantizedWeight")
      .Input("QuantizedBias")
      .Output("QuantizedOutput")
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.Setup(DeviceType::CPU);
  Tensor *eq_output = net.GetTensor("ExpectedQuantizedOutput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  net.Run();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}
}  // namespace

TEST_F(FullyConnectedOpTest, Quant) {
  QuantRandom(1, 16, 16, 32, 16);
  QuantRandom(1, 7, 7, 32, 16);
  QuantRandom(1, 7, 7, 512, 128);
  QuantRandom(1, 1, 1, 2048, 1024);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
