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

class ChannelShuffleOpTest : public OpsTestBase {};

TEST_F(ChannelShuffleOpTest, C8G4_CPU) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 1, 2, 8},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("group", 4)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 1, 2, 8}, {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ChannelShuffleOpTest, C16G4_OPENCL) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::GPU, float>(
      "Input", {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("group", 4)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 1, 2, 16},
      {0,  4,  8,  12, 1,  5,  9,  13, 2,  6,  10, 14, 3,  7,  11, 15,
       16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
