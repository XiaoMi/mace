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

class SliceOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestSlice(const std::vector<index_t> &input_shape,
               const std::vector<T> &input,
               const int offset,
               const int output_dim,
               const std::vector<index_t> &output_shape,
               const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("Slice", "SliceTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("axes", {-1})
      .AddIntsArg("starts", {offset})
      .AddIntsArg("ends", {offset + output_dim})
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(SliceOpTest, Simple2Dim) {
  TestSlice<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    2, 3, {3, 3},
    {3, 4, 5, 8, 9, 10, 13, 14, 15});
}

TEST_F(SliceOpTest, Simple3Dim) {
  TestSlice<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    1, 2, {2, 3, 2},
    {2, 3, 7, 8, 12, 13, 2, 3, 7, 8, 12, 13});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
