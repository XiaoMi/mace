//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class ReluOpTest : public OpsTestBase {};

template <DeviceType D>
void TestSimple() {
  OpsTestNet net;
  OpDefBuilder("Relu", "ReluTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input",
                                  {2, 2, 2, 2},
                                  {-7, 7, -6, 6, -5, 5, -4, 4,
                                   -3, 3, -2, 2, -1, 1, 0, 0});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({2, 2, 2, 2},
                                      {0, 7, 0, 6, 0, 5, 0, 4,
                                       0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ReluOpTest, CPUSimple) {
  TestSimple<DeviceType::CPU>();
}

TEST_F(ReluOpTest, NEONSimple) {
  TestSimple<DeviceType::NEON>();
}

TEST_F(ReluOpTest, OPENCLSimple) {
  TestSimple<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestUnalignedSimple() {
  OpsTestNet net;
  OpDefBuilder("Relu", "ReluTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input",
                                  {1, 1, 3, 2},
                                  {-7, 7, -6, 6, -5, 5});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({1, 1, 3, 2},
                                      {0, 7, 0, 6, 0, 5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ReluOpTest, CPUUnalignedSimple) {
  TestUnalignedSimple<DeviceType::CPU>();
}

TEST_F(ReluOpTest, NEONUnalignedSimple) {
  TestUnalignedSimple<DeviceType::NEON>();
}

TEST_F(ReluOpTest, OPENCLUnalignedSimple) {
  TestUnalignedSimple<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestSimpleReluX() {
  OpsTestNet net;
  OpDefBuilder("Relu", "ReluTest")
      .Input("Input")
      .Output("Output")
      .AddFloatArg("max_limit", 6)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input",
                                  {2, 2, 2, 2},
                                  {-7, 7, -6, 6, -5, 5, -4, 4,
                                   -3, 3, -2, 2, -1, 1, 0, 0});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({2, 2, 2, 2},
                                      {0, 6, 0, 6, 0, 5, 0, 4,
                                       0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ReluOpTest, CPUSimpleReluX) {
  TestSimpleReluX<DeviceType::CPU>();
}

TEST_F(ReluOpTest, NEONSimpleReluX) {
  TestSimpleReluX<DeviceType::NEON>();
}

TEST_F(ReluOpTest, OPENCLSimpleReluX) {
  TestSimpleReluX<DeviceType::OPENCL>();
}

template <DeviceType D>
void TestUnalignedSimpleReluX() {
  OpsTestNet net;
  OpDefBuilder("Relu", "ReluTest")
      .Input("Input")
      .Output("Output")
      .AddFloatArg("max_limit", 6)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input",
                                  {1, 1, 1, 7},
                                  {-7, 7, -6, 6, -5, 5, -4});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({1, 1, 1, 7},
                                      {0, 6, 0, 6, 0, 5, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ReluOpTest, CPUUnalignedSimpleReluX) {
  TestUnalignedSimpleReluX<DeviceType::CPU>();
}

TEST_F(ReluOpTest, NEONUnalignedSimpleReluX) {
  TestUnalignedSimpleReluX<DeviceType::NEON>();
}

TEST_F(ReluOpTest, OPENCLUnalignedSimpleReluX) {
  TestUnalignedSimpleReluX<DeviceType::OPENCL>();
}

}  // namespace mace
