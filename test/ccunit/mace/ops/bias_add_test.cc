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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class BiasAddOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void BiasAddSimple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<D, float>("Bias", {1}, {0.5f}, true);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("InputNCHW")
        .Input("Bias")
        .AddIntArg("has_data_format", 1)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("Input")
        .Input("Bias")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 6, 2, 1},
      {5.5, 5.5, 7.5, 7.5, 9.5, 9.5, 11.5, 11.5, 13.5, 13.5, 15.5, 15.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

template <DeviceType D>
void BiasAddSimple2D() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {2, 3, 1, 2},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<D, float>("Bias", {2, 2},
                                  {0.1f, 0.2f, 0.3f, 0.4f}, true);
  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("InputNCHW")
        .Input("Bias")
        .AddIntArg("has_data_format", 1)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("Input")
        .Input("Bias")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = net.CreateTensor<float>(
      {2, 3, 1, 2},
      {5.1, 5.2, 7.1, 7.2, 9.1, 9.2, 11.3, 11.4, 13.3, 13.4, 15.3, 15.4});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

}  // namespace

TEST_F(BiasAddOpTest, BiasAddSimpleCPU) { BiasAddSimple<DeviceType::CPU>(); }

TEST_F(BiasAddOpTest, BiasAddSimple2DCPU) {
  BiasAddSimple2D<DeviceType::CPU>();
}

TEST_F(BiasAddOpTest, BiasAddSimpleOPENCL) {
  BiasAddSimple<DeviceType::GPU>();
}

TEST_F(BiasAddOpTest, SimpleRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64 + rand_r(&seed) % 50;
  index_t width = 64 + rand_r(&seed) % 50;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Bias", {channels}, true, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputNCHW")
      .Input("Bias")
      .AddIntArg("has_data_format", 1)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on gpu
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("Input")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(BiasAddOpTest, ComplexRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103 + rand_r(&seed) % 100;
  index_t width = 113 + rand_r(&seed) % 100;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Bias", {channels}, true, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputNCHW")
      .Input("Bias")
      .AddIntArg("has_data_format", 1)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on gpu
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("Input")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
void TestQuantized(const bool batched_bias,
                   const bool has_data_format) {
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64 + rand_r(&seed) % 50;
  index_t width = 64 + rand_r(&seed) % 50;

  OpsTestNet net;
  std::vector<index_t> input_shape{batch, height, width, channels};
  net.AddRandomInput<CPU, float>("Input", input_shape, false, false);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  if (batched_bias) {
    net.AddRandomInput<CPU, float>("Bias", {batch, channels}, true);
  } else {
    net.AddRandomInput<CPU, float>("Bias", {channels}, true);
  }

  net.AddRandomInput<DeviceType::CPU, float>(
      "OutputNCHW", input_shape, false, true, true);
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputNCHW")
      .Input("Bias")
      .Output("OutputNCHW")
      .AddIntArg("has_data_format", has_data_format)
      .AddIntArg("T", DT_FLOAT)
      .Finalize(net.NewOperatorDef());

  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeBias")
      .Input("Bias")
      .Output("QuantizedBias")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
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

  net.AddRandomInput<DeviceType::CPU, uint8_t>("QuantizedOutput", input_shape);
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("QuantizedInput")
      .Input("QuantizedBias")
      .Output("QuantizedOutput")
      .AddIntArg("has_data_format", has_data_format)
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

TEST_F(BiasAddOpTest, Quantized) {
  TestQuantized(false, false);
  TestQuantized(false, true);
  TestQuantized(true, false);
  TestQuantized(true, true);
}

#ifdef MACE_ENABLE_BFLOAT16
TEST_F(BiasAddOpTest, BFloat16) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103 + rand_r(&seed) % 100;
  index_t width = 113 + rand_r(&seed) % 100;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input",
                                             {batch, channels, height, width});
  net.AddRandomInput<DeviceType::CPU, float>("Bias", {channels}, true);

  net.Cast<DeviceType::CPU, float, BFloat16>("Input", "BF16Input");
  net.Cast<DeviceType::CPU, float, BFloat16>("Bias", "BF16Bias");

  // Construct graph
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("Input")
      .Input("Bias")
      .AddIntArg("has_data_format", 1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DT_FLOAT))
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("BiasAdd", "BF16BiasAddTest")
      .Input("BF16Input")
      .Input("BF16Bias")
      .AddIntArg("has_data_format", 1)
      .Output("BF16Output")
      .AddIntArg("T", static_cast<int>(DT_BFLOAT16))
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  net.Cast<DeviceType::CPU, BFloat16, float>("BF16Output", "CastOutput");

  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("CastOutput"), 1e-5);
}
#endif  // MACE_ENABLE_BFLOAT16

}  // namespace test
}  // namespace ops
}  // namespace mace
