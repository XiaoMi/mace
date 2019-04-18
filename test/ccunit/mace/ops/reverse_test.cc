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

class ReverseOpTest : public OpsTestBase {};

namespace {

void TestReverse(const std::vector<index_t> &input_shape,
                 const std::vector<float> &input,
                 const std::vector<index_t> &axis_shape,
                 const std::vector<int32_t> &axis,
                 const std::vector<float> &outputs) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);
  net.AddInputFromArray<CPU, int32_t>("Axis", axis_shape, axis);

  OpDefBuilder("Reverse", "ReverseOpTest")
      .Input("Input")
      .Input("Axis")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, float>("ExpectedOutput", input_shape,
                                    outputs);
  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}

}  // namespace

TEST_F(ReverseOpTest, SimpleCPU) {
  TestReverse({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1}, {0},
      {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5});
  TestReverse({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1}, {1},
      {4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7});
  TestReverse({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1}, {2},
      {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
