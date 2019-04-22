// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ActivationOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void TestSimpleRelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "ReluTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "RELU")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2}, {0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimpleRelu) { TestSimpleRelu<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimpleRelu) {
  TestSimpleRelu<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestSimpleLeakyRelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "ReluTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "LEAKYRELU")
      .AddFloatArg("leakyrelu_coefficient", 0.1)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2},
      {-0.7, 7, -0.6, 6, -0.5, 5, -0.4, 4, -0.3, 3, -0.2, 2, -0.1, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimpleLeakyRelu) {
  TestSimpleLeakyRelu<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLSimpleLeakyRelu) {
  TestSimpleLeakyRelu<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestUnalignedSimpleRelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 3, 2, 1}, {-7, 7, -6, 6, -5, 5});

  OpDefBuilder("Activation", "ReluTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "RELU")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>({1, 3, 2, 1}, {0, 7, 0, 6, 0, 5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUUnalignedSimpleRelu) {
  TestUnalignedSimpleRelu<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLUnalignedSimpleRelu) {
  TestUnalignedSimpleRelu<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestSimpleRelux() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "ReluxTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "RELUX")
      .AddFloatArg("max_limit", 6)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2}, {0, 6, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimple) { TestSimpleRelux<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimple) { TestSimpleRelux<DeviceType::GPU>(); }

namespace {
template <DeviceType D>
void TestSimpleReluRelux() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "ReluxTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "RELUX")
      .AddFloatArg("max_limit", 6)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2}, {0, 6, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimpleRelux) {
  TestSimpleReluRelux<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLSimpleRelux) {
  TestSimpleReluRelux<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestSimplePrelu() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, -5, -4, -4, -3, 3, -2, 2, -1, -1, 0, 0});
  net.AddInputFromArray<D, float>("Alpha", {2}, {2.0, 3.0}, true);

  if (D == DeviceType::GPU) {
    OpDefBuilder("Activation", "PreluTest")
        .Input("Input")
        .Input("Alpha")
        .Output("Output")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("Activation", "PreluTest")
        .Input("InputNCHW")
        .Input("Alpha")
        .Output("OutputNCHW")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2},
      {-14, 7, -12, 6, -10, -15, -8, -12, -6, 3, -4, 2, -2, -3, 0, 0});
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimplePrelu) { TestSimplePrelu<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimplePrelu) {
  TestSimplePrelu<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestSimpleTanh() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "TanhTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "TANH")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2},
      {-0.99999834, 0.99999834, -0.99998771, 0.99998771, -0.9999092, 0.9999092,
       -0.9993293, 0.9993293, -0.99505475, 0.99505475, -0.96402758, 0.96402758,
       -0.76159416, 0.76159416, 0., 0.});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimpleTanh) { TestSimpleTanh<DeviceType::CPU>(); }

TEST_F(ActivationOpTest, OPENCLSimpleTanh) {
  TestSimpleTanh<DeviceType::GPU>();
}

namespace {
template <DeviceType D>
void TestSimpleSigmoid() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});

  OpDefBuilder("Activation", "SigmoidTest")
      .Input("Input")
      .Output("Output")
      .AddStringArg("activation", "SIGMOID")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>(
      {2, 2, 2, 2},
      {9.11051194e-04, 9.99088949e-01, 2.47262316e-03, 9.97527377e-01,
       6.69285092e-03, 9.93307149e-01, 1.79862100e-02, 9.82013790e-01,
       4.74258732e-02, 9.52574127e-01, 1.19202922e-01, 8.80797078e-01,
       2.68941421e-01, 7.31058579e-01, 5.00000000e-01, 5.00000000e-01});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ActivationOpTest, CPUSimpleSigmoid) {
  TestSimpleSigmoid<DeviceType::CPU>();
}

TEST_F(ActivationOpTest, OPENCLSimpleSigmoid) {
  TestSimpleSigmoid<DeviceType::GPU>();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
