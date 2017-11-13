//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class AddnOpTest : public OpsTestBase {};

template<DeviceType D>
void SimpleAdd2() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("AddN", "AddNTest")
      .Input("Input1")
      .Input("Input2")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input1", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input2", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({1, 1, 2, 3}, {2, 4, 6, 8, 10, 12});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(AddnOpTest, CPUSimpleAdd2) {
  SimpleAdd2<DeviceType::CPU>();
}

TEST_F(AddnOpTest, NEONSimpleAdd2) {
  SimpleAdd2<DeviceType::NEON>();
}

TEST_F(AddnOpTest, OPENCLSimpleAdd2) {
  SimpleAdd2<DeviceType::OPENCL>();
}

template<DeviceType D>
void SimpleAdd3() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("AddN", "AddNTest")
      .Input("Input1")
      .Input("Input2")
      .Input("Input3")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input1", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input2", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input3", {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});

  // Run
  net.RunOp(D);

  auto expected = CreateTensor<float>({1, 1, 2, 3}, {3, 6, 9, 12, 15, 18});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(AddnOpTest, CPUSimpleAdd3) {
  SimpleAdd3<DeviceType::CPU>();
}

TEST_F(AddnOpTest, NEONSimpleAdd3) {
  SimpleAdd3<DeviceType::NEON>();
}

template<DeviceType D>
void RandomTest() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("AddN", "AddNTest")
      .Input("Input1")
      .Input("Input2")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input1", {1, 2, 3, 4});
  net.AddRandomInput<D, float>("Input2", {1, 2, 3, 4});

  // Check
  net.RunOp(D);

  Tensor result;
  result.Copy(*net.GetOutput("Output"));

  // Run
  net.RunOp();

  ExpectTensorNear<float>(*net.GetOutput("Output"), result, 1e-5);
}

TEST_F(AddnOpTest, CPURandom) {
  RandomTest<DeviceType::CPU>();
}

TEST_F(AddnOpTest, NEONRandom) {
  RandomTest<DeviceType::NEON>();
}

TEST_F(AddnOpTest, OPENCLRandom) {
  RandomTest<DeviceType::OPENCL>();
}

}  // namespace mace
