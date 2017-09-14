//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class BatchNormOpTest : public OpsTestBase {};

TEST_F(BatchNormOpTest, SimpleCPU) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
        .Input("Input")
        .Input("Scale")
        .Input("Offset")
        .Input("Mean")
        .Input("Var")
        .Output("Output")
        .Finalize(net.operator_def());

  // Add input data
  net.AddInputFromArray<float>("Input", {1, 1, 6, 2},
                    {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<float>("Scale", {1}, {4.0f});
  net.AddInputFromArray<float>("Offset", {1}, {2.0});
  net.AddInputFromArray<float>("Mean", {1}, {10});
  net.AddInputFromArray<float>("Var", {1}, {11.67f});

  // Run
  net.RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 6, 2},
                                        {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                         3.17, 3.17, 5.51, 5.51, 7.86, 7.86});

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.01);
}

TEST_F(BatchNormOpTest, SimpleNeon) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 10 + rand() % 50;
  index_t width = 10 + rand() % 50;
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
          .Input("Input")
          .Input("Scale")
          .Input("Offset")
          .Input("Mean")
          .Input("Var")
          .Output("Output")
          .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<float>("Input", {batch, channels, height, width});
  net.AddRandomInput<float>("Scale", {channels});
  net.AddRandomInput<float>("Offset", {channels});
  net.AddRandomInput<float>("Mean", {channels});
  net.AddRandomInput<float>("Var", {channels}, true);

  // run cpu
  net.RunOp();

  // Check
  Tensor expected = *net.GetOutput("Output");

  // Run NEON
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-5);
}

}
