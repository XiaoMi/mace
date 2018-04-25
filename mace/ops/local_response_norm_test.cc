//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class LocalResponseNormOpTest : public OpsTestBase {};

template<DeviceType D>
void Simple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 1, 2, 6},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>("Input", NHWC, "InputNCHW", NCHW);

    OpDefBuilder("LocalResponseNorm", "LocalResponseNormTest")
      .Input("InputNCHW")
      .AddIntArg("depth_radius", 5)
      .AddFloatArg("bias", 1.0f)
      .AddFloatArg("alpha", 1.0f)
      .AddFloatArg("beta", 0.5f)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>("OutputNCHW", NCHW, "Output", NHWC);
  }

  // Check
  auto expected =
    CreateTensor<float>({1, 1, 2, 6}, {0.28, 0.28, 0.39, 0.39, 0.51, 0.51,
                                       0.34, 0.34, 0.40, 0.40, 0.47, 0.47});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0, 1e-2);
}

TEST_F(LocalResponseNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

}  // namespace test
}  // namespace ops
}  // namespace mace
