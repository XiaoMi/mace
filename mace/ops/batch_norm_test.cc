//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class BatchNormOpTest : public OpsTestBase {};

template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 1, 6, 2},
                               {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<D, float>("Scale", {1}, {4.0f});
  net.AddInputFromArray<D, float>("Offset", {1}, {2.0});
  net.AddInputFromArray<D, float>("Mean", {1}, {10});
  net.AddInputFromArray<D, float>("Var", {1}, {11.67f});
  net.AddInputFromArray<D, float>("Epsilon", {}, {1e-3});

  // Run
  net.RunOp(D);

  // Check
  auto expected =
      CreateTensor<float>({1, 1, 6, 2}, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                         3.17, 3.17, 5.51, 5.51, 7.86, 7.86});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(BatchNormOpTest, SimpleCPU) {
  Simple<DeviceType::CPU>();
}

TEST_F(BatchNormOpTest, SimpleNEON) {
  Simple<DeviceType::NEON>();
}

TEST_F(BatchNormOpTest, SimpleOPENCL) {
  Simple<DeviceType::OPENCL>();
}

TEST_F(BatchNormOpTest, SimpleRandomNeon) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 64;
  index_t width = 64;
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", {batch, channels, height, width});
  net.AddRandomInput<DeviceType::CPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::CPU, float>("Epsilon", {}, {1e-3});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run NEON
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(BatchNormOpTest, ComplexRandomNeon) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 103;
  index_t width = 113;
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", {batch, channels, height, width});
  net.AddRandomInput<DeviceType::CPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::CPU, float>("Epsilon", {}, {1e-3});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run NEON
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(BatchNormOpTest, SimpleRandomOPENCL) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 64;
  index_t width = 64;
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>("Input", {batch, channels, height, width});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::OPENCL, float>("Epsilon", {}, {1e-3});

  // Run NEON
  net.RunOp(DeviceType::OPENCL);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // run cpu
  net.RunOp();

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(BatchNormOpTest, ComplexRandomOPENCL) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 103;
  index_t width = 113;
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>("Input", {batch, channels, height, width});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::OPENCL, float>("Epsilon", {}, {1e-3});

  // Run NEON
  net.RunOp(DeviceType::OPENCL);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // run cpu
  net.RunOp();

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}

}
