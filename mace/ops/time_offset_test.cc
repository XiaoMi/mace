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

class TimeOffsetOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestTimeOffset(const std::vector<index_t> &input_shape,
                    const std::vector<T> &input,
                    const int offset,
                    const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("TimeOffset", "TimeOffsetTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("offset", offset)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", input_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(TimeOffsetOpTest, Simple2Dim) {
  TestTimeOffset<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    -2,
    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5});

  TestTimeOffset<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    -1,
    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  TestTimeOffset<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    0,
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  TestTimeOffset<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    1,
    {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15});

  TestTimeOffset<DeviceType::CPU, float>(
    {3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    2,
    {11, 12, 13, 14, 15, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15});
}


TEST_F(TimeOffsetOpTest, Simple3Dim) {
  TestTimeOffset<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    -2,
    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
     1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5});

  TestTimeOffset<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    -1,
    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  TestTimeOffset<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    0,
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  TestTimeOffset<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    1,
    {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
     6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15});

  TestTimeOffset<DeviceType::CPU, float>(
    {2, 3, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    2,
    {11, 12, 13, 14, 15, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
     11, 12, 13, 14, 15, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
