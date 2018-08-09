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

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class SoftmaxOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 1, 2, 4},
                                  {1, 1, 1, 1, 1, 2, 3, 4});
  auto expected = CreateTensor<float>(
      {1, 1, 2, 4},
      {0.25, 0.25, 0.25, 0.25, 0.0320586, 0.08714432, 0.23688282, 0.64391426});

  if (D == DeviceType::CPU) {
    // test 4d softmax
    net.TransformDataFormat<CPU, float>("Input", NHWC, "InputNCHW", NCHW);
    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.TransformDataFormat<CPU, float>("OutputNCHW", NCHW, "Output", NHWC);

    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);

    // check 2d softmax
    net.AddInputFromArray<D, float>("Input2d", {2, 4},
                                    {1, 1, 1, 1, 1, 2, 3, 4});
    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("Input2d")
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.GetOutput("Output")->Reshape({1, 1, 2, 4});
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("InputImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);

    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
}  // namespace

TEST_F(SoftmaxOpTest, CPUSimple) { Simple<DeviceType::CPU>(); }
TEST_F(SoftmaxOpTest, OPENCLSimple) { Simple<DeviceType::GPU>(); }

namespace {
template <DeviceType D>
void Complex(const std::vector<index_t> &logits_shape) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", logits_shape);

  if (logits_shape.size() == 4) {
    net.TransformDataFormat<CPU, float>("Input", NHWC, "InputNCHW", NCHW);

    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("Input")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  }
  // Run on cpu
  net.RunOp();

  if (logits_shape.size() == 4) {
    net.TransformDataFormat<CPU, float>("OutputNCHW", NCHW, "Output", NHWC);
  }

  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<D, float>(&net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);

  OpDefBuilder("Softmax", "SoftmaxTest")
      .Input("InputImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on gpu
  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, float>(&net, "OutputImage", "OPENCLOutput",
                          kernels::BufferType::IN_OUT_CHANNEL);

  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5);
}
}  // namespace

TEST_F(SoftmaxOpTest, OPENCLAligned) {
  Complex<DeviceType::GPU>({1, 256, 256, 3});
  Complex<DeviceType::GPU>({1, 128, 128, 16});
}

TEST_F(SoftmaxOpTest, OPENCLMulBatchAligned) {
  Complex<DeviceType::GPU>({5, 64, 64, 3});
  Complex<DeviceType::GPU>({8, 128, 128, 8});
}

TEST_F(SoftmaxOpTest, OPENCLUnAligned) {
  Complex<DeviceType::GPU>({1, 113, 107, 13});
  Complex<DeviceType::GPU>({5, 211, 107, 1});
}

TEST_F(SoftmaxOpTest, OPENCLAlignedRank2) {
  Complex<DeviceType::GPU>({1, 1001});
  Complex<DeviceType::GPU>({3, 1001});
}

namespace {

void TestQuantizedSoftmax(const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input", input_shape, false, true);

  OpDefBuilder("Softmax", "SoftmaxTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  net.RunOp();
  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();
  OpDefBuilder("Softmax", "SoftmaxQuantizeTest")
      .Input("QuantizedInput")
      .Output("QuantizedOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.Setup(DeviceType::CPU);
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(1.0f / 255);
  q_output->SetZeroPoint(0);
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
                             *net.GetTensor("DequantizedOutput"), 0.1);
}

}  // namespace

TEST_F(SoftmaxOpTest, QuantizeTest) {
  TestQuantizedSoftmax({5, 10});
  TestQuantizedSoftmax({50, 100});
  TestQuantizedSoftmax({1, 31});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
