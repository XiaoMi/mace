//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/resize_bilinear.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ResizeBilinearTest : public OpsTestBase {};

TEST_F(ResizeBilinearTest, CPUResizeBilinearWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 2, 4}, input);
  net.AddInputFromArray<DeviceType::CPU, int>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 2}, {0, 2, 8, 10, 16, 18});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(ResizeBilinearTest, ResizeBilinearWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .AddIntArg("align_corners", 1)
      .Finalize(net.NewOperatorDef());

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 2, 4}, input);
  net.AddInputFromArray<DeviceType::CPU, int>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 2}, {0, 3, 8, 11, 16, 19});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

template <DeviceType D>
void TestRandomResizeBilinear() {
  srand(time(nullptr));
  testing::internal::LogToStderr();
  for (int round = 0; round < 10; ++round) {
    int batch = 1 + rand() % 5;
    int channels = 1 + rand() % 100;
    int height = 1 + rand() % 100;
    int width = 1 + rand() % 100;
    int in_height = 1 + rand() % 100;
    int in_width = 1 + rand() % 100;

    // Construct graph
    OpsTestNet net;
    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("Input")
        .Input("OutSize")
        .Output("Output")
        .AddIntArg("align_corners", 1)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, channels, in_height, in_width});
    net.AddInputFromArray<D, int>("OutSize", {2}, {height, width});

    // Run
    net.RunOp(D);
    Tensor actual;
    actual.Copy(*net.GetOutput("Output"));

    // Run on CPU
    net.RunOp(DeviceType::CPU);
    Tensor *expected = net.GetOutput("Output");

    // Check
    ExpectTensorNear<float>(*expected, actual, 0.001);
  }
}

TEST_F(ResizeBilinearTest, NEONRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::NEON>();
}

TEST_F(ResizeBilinearTest, OPENCLRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::OPENCL>();
}
