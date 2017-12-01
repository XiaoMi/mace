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
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.AddInputFromArray<DeviceType::CPU, int>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 6, 7, 8});

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
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.AddInputFromArray<DeviceType::CPU, int>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 9, 10, 11});

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
    int align_corners = rand() % 1;

    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, in_height, in_width, channels});
    net.AddInputFromArray<D, int>("OutSize", {2}, {height, width});

    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .AddIntArg("align_corners", align_corners)
      .AddIntsArg("size", {height, width})
      .Finalize(net.NewOperatorDef());
    // Run on CPU
    net.RunOp(DeviceType::CPU);
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    if (D == DeviceType::OPENCL) {
      BufferToImage<D, float>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);

      OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("InputImage")
        .Input("OutSize")
        .Output("OutputImage")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());
      // Run
      net.RunOp(D);

      ImageToBuffer<D, float>(net, "OutputImage", "DeviceOutput", kernels::BufferType::IN_OUT);
    } else {
      // TODO support NEON
    }
    // Check
    ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 0.001);
  }
}

/*
TEST_F(ResizeBilinearTest, NEONRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::NEON>();
}
*/

TEST_F(ResizeBilinearTest, OPENCLRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::OPENCL>();
}
