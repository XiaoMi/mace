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

class PadContextOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestPadContext(const std::vector<index_t> &input_shape,
               const std::vector<T> &input,
               const int left_context,
               const int right_context,
               const std::vector<index_t> &output_shape,
               const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("PadContext", "PadContextTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("left_context", left_context)
      .AddIntArg("right_context", right_context)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  auto expected = net.CreateTensor<T>(output_shape, output);
  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(PadContextOpTest, Simple2Dim) {
  TestPadContext<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    2, 3, {8, 5},
    {1, 2, 3, 4, 5,
     1, 2, 3, 4, 5,
     1, 2, 3, 4, 5,
     6, 7, 8, 9, 10,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15});
}

TEST_F(PadContextOpTest, Simple3Dim) {
  TestPadContext<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    1, 2, {2, 6, 5},
    {1, 2, 3, 4, 5,
     1, 2, 3, 4, 5,
     6, 7, 8, 9, 10,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15,
     1, 2, 3, 4, 5,
     1, 2, 3, 4, 5,
     6, 7, 8, 9, 10,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15,
     11, 12, 13, 14, 15});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
