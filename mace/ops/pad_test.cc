//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class PadTest : public OpsTestBase {};

template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<D, float>("Input", {1, 2, 3, 1}, 2);
  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Pad", "PadTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddIntsArg("paddings", {0, 0, 1, 2, 1, 2, 0, 0})
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Pad", "PadTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("paddings", {0, 0, 1, 2, 1, 2, 0, 0})
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp();
  }

  auto output = net.GetTensor("Output");

  auto expected = CreateTensor<float>({1, 5, 6, 1},
                                      {
                                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                          1.0, 2, 2, 2, 1.0, 1.0,
                                          1.0, 2, 2, 2, 1.0, 1.0,
                                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}

TEST_F(PadTest, SimpleCPU) {
  Simple<DeviceType::CPU>();
}

TEST_F(PadTest, SimpleGPU) {
  Simple<DeviceType::OPENCL>();
}

TEST_F(PadTest, ComplexCPU) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<DeviceType::CPU, float>("Input", {1, 1, 1, 2}, 2);
  OpDefBuilder("Pad", "PadTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("paddings", {0, 0, 1, 1, 1, 1, 1, 1})
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");

  auto expected = CreateTensor<float>(
      {1, 3, 3, 4},
      {
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}

template <typename T>
void Complex(const std::vector<index_t> &input_shape,
             const std::vector<int> &paddings) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>("Input", input_shape);

  OpDefBuilder("Pad", "PadTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("paddings", paddings)
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<DeviceType::OPENCL, T>(&net, "Input", "InputImage",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("Pad", "PadTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("paddings", paddings)
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(&net, "OutputImage", "OpenCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);

  auto output = net.GetTensor("OpenCLOutput");

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(expected, *output, 1e-1);
  } else {
    ExpectTensorNear<float>(expected, *output, 1e-5);
  }
}

TEST_F(PadTest, ComplexFloat) {
  Complex<float>({1, 32, 32, 4}, {0, 0, 2, 2, 1, 1, 0, 0});
  Complex<float>({1, 31, 37, 16}, {0, 0, 2, 0, 1, 0, 0, 0});
  Complex<float>({1, 128, 128, 32}, {0, 0, 0, 1, 0, 2, 0, 0});
}

TEST_F(PadTest, ComplexHalf) {
  Complex<half>({1, 32, 32, 4}, {0, 0, 2, 2, 1, 1, 0, 0});
  Complex<half>({1, 31, 37, 16}, {0, 0, 2, 0, 1, 0, 0, 0});
  Complex<half>({1, 128, 128, 32}, {0, 0, 0, 1, 0, 2, 0, 0});
}

}  // namespace test
}  // namespace ops
}  // namespace mace

