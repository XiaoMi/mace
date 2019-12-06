// Copyright 2019 The MACE Authors. All Rights Reserved.
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

class SelectOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestSelect(const std::vector<index_t> &input_shape,
                const std::vector<uint8_t> &input,
                const std::vector<index_t> &x_shape,
                const std::vector<T> &x,
                const std::vector<index_t> &y_shape,
                const std::vector<T> &y,
                const std::vector<index_t> &output_shape,
                const std::vector<T> &output) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder builder("Select", "SelectTest");
  builder.Input("Input");
  if (x.size() > 0) {
    builder.Input("X").Input("Y");
  }
  builder.Output("Output").Finalize(net.NewOperatorDef());

  net.AddInputFromArray<D, uint8_t>(MakeString("Input"), input_shape, input);
  if (x.size() > 0) {
    net.AddInputFromArray<D, T>(MakeString("X"), x_shape, x);
    net.AddInputFromArray<D, T>(MakeString("Y"), y_shape, y);
  }

  // Run
  net.RunOp();

  net.AddInputFromArray<D, T>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace


TEST_F(SelectOpTest, SimpleTestWithData) {
  TestSelect<DeviceType::CPU, float>(
    {2, 3},
    {true, false, false, false, true, true},
    {2, 3},
    {3.0, 2.0, 3.0, 4.0, 5.0, 6.0},
    {2, 3},
    {3.0, -1.0, -2.0, -3.0, 8.0, 9.0},
    {2, 3},
    {3.0, -1.0, -2.0, -3.0, 5.0, 6.0});
}

TEST_F(SelectOpTest, SimpleTestWithDataBroadcast) {
  TestSelect<DeviceType::CPU, float>(
    {2},
    {true, false},
    {2, 3},
    {3.0, 2.0, 3.0, 4.0, 5.0, 6.0},
    {2, 3},
    {3.0, -1.0, -2.0, -3.0, 8.0, 9.0},
    {2, 3},
    {3, 2, 3, -3, 8, 9});
}

TEST_F(SelectOpTest, SimpleTestWithNoDataBroadcast1) {
  TestSelect<DeviceType::CPU, float>(
    {2},
    {true, false},
    {}, {}, {}, {},
    {1, 1},
    {0});
}

TEST_F(SelectOpTest, SimpleTestWithNoDataBroadcast2) {
  TestSelect<DeviceType::CPU, float>(
    {2, 3},
    {true, false, false, false, true, true},
    {}, {}, {}, {},
    {3, 2},
    {0, 0, 1, 1, 1, 2});
}

TEST_F(SelectOpTest, SimpleTestWithNoDataBroadcast3) {
  TestSelect<DeviceType::CPU, float>(
    {2, 2, 3},
    {true, false, false, false, true, true,
     true, false, false, false, true, true},
    {}, {}, {}, {},
    {6, 3},
    {0, 0, 0, 0, 1, 1, 0, 1, 2,
     1, 0, 0, 1, 1, 1, 1, 1, 2});
}

TEST_F(SelectOpTest, SimpleTestWithNoDataBroadcast4) {
  TestSelect<DeviceType::CPU, float>(
    {2, 2, 2, 3},
    {true, false, false, false, true, true,
     true, false, false, false, true, true,
     true, false, false, false, true, true,
     true, false, false, false, true, true},
    {}, {}, {}, {},
    {12, 4},
    {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2,
     0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 2,
     1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 2,
     1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2});
}

TEST_F(SelectOpTest, SimpleTestWithNoDataBroadcast5) {
  TestSelect<DeviceType::CPU, float>(
    {2, 2, 2, 2, 3},
    {true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true,
    true, false, false, false, true, true},
    {}, {}, {}, {},
    {24, 5},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2,
     0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2,
     0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 2,
     0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2,
     1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 2,
     1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 2,
     1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2,
     1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
