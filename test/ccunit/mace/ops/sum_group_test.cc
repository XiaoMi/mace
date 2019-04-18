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

class SumGroupOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestSumGroup(const std::vector<index_t> &input_shape,
                  const std::vector<T> &input,
                  const std::vector<int> &sizes,
                  const std::vector<index_t> &output_shape,
                  const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);
  const index_t output_dim = sizes.size();
  net.AddInputFromArray<CPU, int>(MakeString("Sizes"),
                                  {output_dim},
                                  sizes);

  OpDefBuilder("SumGroup", "SumGroupTest")
      .Input("Input")
      .Input("Sizes")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(SumGroupOpTest, SimpleTest) {
  TestSumGroup<DeviceType::CPU, float>(
    {1, 5, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    {2, 1, 2, 3, 2},
    {1, 5, 5},
    {3, 3, 9, 21, 19,
     5, 4, 11, 24, 21,
     7, 5, 13, 27, 23,
     9, 6, 15, 30, 25,
     11, 7, 17, 33, 27});
}
}  // namespace test
}  // namespace ops
}  // namespace mace
