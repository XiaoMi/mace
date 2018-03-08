//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class SoftmaxOpTest : public OpsTestBase {};

template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 1, 2, 4},
                                  {1, 1, 1, 1, 1, 2, 3, 4});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("InputImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Softmax", "SoftmaxTest")
        .Input("Input")
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {1, 1, 2, 4},
      {0.25, 0.25, 0.25, 0.25, 0.0320586, 0.08714432, 0.23688282, 0.64391426});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-7);
}

TEST_F(SoftmaxOpTest, CPUSimple) { Simple<DeviceType::CPU>(); }
TEST_F(SoftmaxOpTest, OPENCLSimple) { Simple<DeviceType::OPENCL>(); }

template <DeviceType D>
void Complex(const std::vector<index_t> &logits_shape) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", logits_shape);

  OpDefBuilder("Softmax", "SoftmaxTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run on cpu
  net.RunOp();
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);

  OpDefBuilder("Softmax", "SoftmaxTest")
      .Input("InputImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on gpu
  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, float>(net, "OutputImage", "OPENCLOutput",
                          kernels::BufferType::IN_OUT_CHANNEL);

  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5);
}

TEST_F(SoftmaxOpTest, OPENCLAligned) {
  Complex<DeviceType::OPENCL>({1, 256, 256, 3});
  Complex<DeviceType::OPENCL>({1, 128, 128, 16});
}

TEST_F(SoftmaxOpTest, OPENCLMulBatchAligned) {
  Complex<DeviceType::OPENCL>({5, 64, 64, 3});
  Complex<DeviceType::OPENCL>({8, 128, 128, 8});
}

TEST_F(SoftmaxOpTest, OPENCLUnAligned) {
  Complex<DeviceType::OPENCL>({1, 113, 107, 13});
  Complex<DeviceType::OPENCL>({5, 211, 107, 1});
}

}  // namespace mace
