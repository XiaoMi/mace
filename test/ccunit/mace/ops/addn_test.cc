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

class AddnOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void SimpleAdd2() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("AddN", "AddNTest")
      .Input("Input1")
      .Input("Input2")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input1", {1, 2, 3, 1}, {1, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input2", {1, 2, 3, 1}, {1, 2, 3, 4, 5, 6});

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>({1, 2, 3, 1}, {2, 4, 6, 8, 10, 12});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(AddnOpTest, CPUSimpleAdd2) { SimpleAdd2<DeviceType::CPU>(); }

namespace {
template <DeviceType D>
void SimpleAdd3() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input0", {1, 2, 3, 1},
                                  {-2.06715, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input1", {1, 2, 3, 1},
                                  {0.875977, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input2", {1, 2, 3, 1},
                                  {1.34866, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input3", {1, 2, 3, 1},
                                  {-0.1582, 2, 3, 4, 5, 6});

  OpDefBuilder("AddN", "AddNTest")
      .Input("Input0")
      .Input("Input1")
      .Input("Input2")
      .Input("Input3")
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);

  auto expected =
      net.CreateTensor<float>({1, 2, 3, 1}, {-0.000713, 8, 12, 16, 20, 24});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4, 1e-3);
}
}  // namespace

TEST_F(AddnOpTest, CPUSimpleAdd3) { SimpleAdd3<DeviceType::CPU>(); }
TEST_F(AddnOpTest, GPUSimpleAdd3) { SimpleAdd3<DeviceType::GPU>(); }

namespace {
template <DeviceType D>
void RandomTest() {
  testing::internal::LogToStderr();
  static unsigned int seed = time(NULL);

  for (int round = 0; round < 10; ++round) {
    // generate random input
    index_t n = 1 + (rand_r(&seed) % 5);
    index_t h = 1 + (rand_r(&seed) % 100);
    index_t w = 1 + (rand_r(&seed) % 100);
    index_t c = 1 + (rand_r(&seed) % 32);
    int input_num = 2 + rand_r(&seed) % 3;
    // Construct graph
    OpsTestNet net;
    auto op_def = OpDefBuilder("AddN", "AddNTest");
    for (int i = 0; i < input_num; ++i) {
      op_def.Input(MakeString("Input", i));
    }
    op_def.Output("Output").Finalize(net.NewOperatorDef());

    // Add input data
    for (int i = 0; i < input_num; ++i) {
      net.AddRandomInput<D, float>(MakeString("Input", i), {n, h, w, c});
    }

    // run on cpu
    net.RunOp();
    // Check
    auto expected = net.CreateTensor<float>();
    expected->Copy(*net.GetOutput("Output"));

    // run on device
    net.RunOp(D);

    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2,
                            1e-2);
  }
}
}  // namespace

TEST_F(AddnOpTest, OPENCLRandom) { RandomTest<DeviceType::GPU>(); }

}  // namespace test
}  // namespace ops
}  // namespace mace
