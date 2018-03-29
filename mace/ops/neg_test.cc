//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class NegOpTest : public OpsTestBase {};

template <DeviceType D>
void NegSimple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    
    OpDefBuilder("Neg", "NegTest")
        .Input("InputImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("Neg", "NegTest")
        .Input("Input")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 6, 2, 1},
      {-5, -5, -7, -7, -9, -9, -11, -11, -13, -13, -15, -15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-8);
}

TEST_F(NegOpTest, NegSimpleCPU) { NegSimple<DeviceType::CPU>(); }

TEST_F(NegOpTest, NegSimpleOPENCL) {
  NegSimple<DeviceType::OPENCL>();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
