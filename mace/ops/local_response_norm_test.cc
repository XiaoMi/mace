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

  OpDefBuilder("LocalResponseNorm", "LocalResponseNormTest")
      .Input("Input")
      .AddIntArg("depth_radius", 5)
      .AddFloatArg("bias", 1.0f)
      .AddFloatArg("alpha", 1.0f)
      .AddFloatArg("beta", 0.5f)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);

  // Check
  auto expected =
    CreateTensor<float>({1, 1, 2, 6}, {0.28, 0.28, 0.39, 0.39, 0.51, 0.51,
                                       0.34, 0.34, 0.40, 0.40, 0.47, 0.47});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0, 1e-2);
}

TEST_F(LocalResponseNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

TEST_F(LocalResponseNormOpTest, NEONTest) {
  srand(time(NULL));
  unsigned int seed;

  // generate random input
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64;
  index_t width = 64;

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("LocalResponseNorm", "LocalResponseNormTest")
      .Input("Input")
      .AddIntArg("depth_radius", 5)
      .AddFloatArg("bias", 1.0f)
      .AddFloatArg("alpha", 1.0f)
      .AddFloatArg("beta", 0.5f)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>(
    "Input", {batch, height, width, channels});

  // run cpu
  net.RunOp();

  OpDefBuilder("LocalResponseNorm", "LocalResponseNormTest")
      .Input("InputNeon")
      .AddIntArg("depth_radius", 5)
      .AddFloatArg("bias", 1.0f)
      .AddFloatArg("alpha", 1.0f)
      .AddFloatArg("beta", 0.5f)
      .Output("OutputNeon")
      .Finalize(net.NewOperatorDef());

  net.FillNHWCInputToNCHWInput<DeviceType::CPU, float>("InputNeon", "Input");

  // Run on neon
  net.RunOp(DeviceType::NEON);
  net.Sync();

  net.FillNHWCInputToNCHWInput<DeviceType::CPU, float>("OutputExpected",
                                                       "Output");

  ExpectTensorNear<float>(*net.GetOutput("OutputExpected"),
                          *net.GetOutput("OutputNeon"),
                          0, 0.001);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
