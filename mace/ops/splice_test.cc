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

class SpliceOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestSplice(const std::vector<index_t> &input_shape,
                const std::vector<T> &input,
                const std::vector<int> &context,
                const int const_dim,
                const std::vector<index_t> &output_shape,
                const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("Splice", "SpliceTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("context", context)
      .AddIntArg("const_component_dim", const_dim)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(SpliceOpTest, WithoutConstDim) {
  TestSplice<DeviceType::CPU, float>(
    {1, 7, 2},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    {-2, -1, 0, 1, 2}, 0,
    {1, 3, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
}

TEST_F(SpliceOpTest, WithConstDim) {
  TestSplice<DeviceType::CPU, float>(
    {1, 5, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    {-2, -1, 0, 1, 2}, 7,
    {1, 1, 22},
    {1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10});
}
}  // namespace test
}  // namespace ops
}  // namespace mace
