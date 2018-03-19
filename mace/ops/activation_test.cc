//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ActivationOpTest : public OpsTestBase {};

template <DeviceType D>
void TestSimpleRelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "ReluTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2}, {0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimpleRelu) { TestSimpleRelu<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimpleRelu) {
  TestSimpleRelu<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestUnalignedSimpleRelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 3, 2, 1}, {-7, 7, -6, 6, -5, 5});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "ReluTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>({1, 3, 2, 1}, {0, 7, 0, 6, 0, 5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUUnalignedSimpleRelu) {
  TestUnalignedSimpleRelu<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLUnalignedSimpleRelu) {
  TestUnalignedSimpleRelu<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimpleRelux() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluxTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "ReluxTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2}, {0, 6, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimple) { TestSimpleRelux<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimple) {
  TestSimpleRelux<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimpleReluRelux() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluxTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "ReluxTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2}, {0, 6, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimpleRelux) {
  TestSimpleReluRelux<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLSimpleRelux) {
  TestSimpleReluRelux<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimplePrelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, -5, -4, -4, -3, 3, -2, 2, -1, -1, 0, 0});
  net.AddInputFromArray<D, float>("Alpha", {2}, {2.0, 3.0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Alpha", "AlphaImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("Activation", "PreluTest")
        .Input("InputImage")
        .Input("AlphaImage")
        .Output("OutputImage")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "PreluTest")
        .Input("Input")
        .Input("Alpha")
        .Output("Output")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2},
      {-14, 7, -12, 6, -10, -15, -8, -12, -6, 3, -4, 2, -2, -3, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimplePrelu) { TestSimplePrelu<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimplePrelu) {
  TestSimplePrelu<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimpleTanh() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "TanhTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "TANH")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "TanhTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "TANH")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2},
      {-0.99999834, 0.99999834, -0.99998771, 0.99998771, -0.9999092, 0.9999092,
       -0.9993293, 0.9993293, -0.99505475, 0.99505475, -0.96402758, 0.96402758,
       -0.76159416, 0.76159416, 0., 0.});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimpleTanh) { TestSimpleTanh<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimpleTanh) {
  TestSimpleTanh<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimpleSigmoid() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "SigmoidTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddStringArg("activation", "SIGMOID")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Activation", "SigmoidTest")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "SIGMOID")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2},
      {9.11051194e-04, 9.99088949e-01, 2.47262316e-03, 9.97527377e-01,
       6.69285092e-03, 9.93307149e-01, 1.79862100e-02, 9.82013790e-01,
       4.74258732e-02, 9.52574127e-01, 1.19202922e-01, 8.80797078e-01,
       2.68941421e-01, 7.31058579e-01, 5.00000000e-01, 5.00000000e-01});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ActivationOpTest, CPUSimpleSigmoid) {
  TestSimpleSigmoid<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLSimpleSigmoid) {
  TestSimpleSigmoid<DeviceType::OPENCL>();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
