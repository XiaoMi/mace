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

class TileOpTest : public OpsTestBase {};

namespace {

void TestTile(const std::vector<index_t> &input_shape,
              const std::vector<float> &input,
              const std::vector<int32_t> &multiples,
              const std::vector<index_t> &output_shape,
              const std::vector<float> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);
  net.AddInputFromArray<CPU, int32_t>(
      "Multiples", {static_cast<int32_t>(multiples.size())}, multiples);

  OpDefBuilder("Tile", "TileOpTest")
      .Input("Input")
      .Input("Multiples")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, float>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}

void TestTileWithDataFormat(const std::vector<index_t> &input_shape,
                            const std::vector<float> &input,
                            const std::vector<int32_t> &multiples,
                            const std::vector<index_t> &output_shape,
                            const std::vector<float> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);
  net.AddInputFromArray<CPU, int32_t>(
      "Multiples", {static_cast<int32_t>(multiples.size())}, multiples);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Tile", "TileOpTest")
      .Input("InputNCHW")
      .Input("Multiples")
      .Output("OutputNCHW")
      .AddIntArg("has_data_format", 1)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  net.AddInputFromArray<CPU, float>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}
}  // namespace

TEST_F(TileOpTest, SimpleTest) {
  TestTile({2, 3}, {0, 1, 2, 3, 4, 5}, {2, 3}, {4, 9},
           {0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5,
            0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5});
  TestTile({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 1, 2},
           {2, 2, 6}, {0, 1, 2, 0, 1, 2, 3, 4,  5,  3, 4,  5,
                       6, 7, 8, 6, 7, 8, 9, 10, 11, 9, 10, 11});
  TestTile({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 1, 2},
           {4, 2, 6}, {0, 1, 2, 0,  1,  2, 3,  4,  5, 3, 4, 5,  6,  7, 8,  6,
                       7, 8, 9, 10, 11, 9, 10, 11, 0, 1, 2, 0,  1,  2, 3,  4,
                       5, 3, 4, 5,  6,  7, 8,  6,  7, 8, 9, 10, 11, 9, 10, 11});
  TestTile({2, 2, 2, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
           {1, 1, 1, 2}, {2, 2, 2, 6},
           {0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  7,  8,  6,
            7,  8,  9,  10, 11, 9,  10, 11, 12, 13, 14, 12, 13, 14, 15, 16,
            17, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22, 23});
  TestTile({2, 2, 2, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
           {1, 2, 2, 1}, {2, 4, 4, 3},
           {0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
            10, 11, 6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  4,  5,  0,  1,
            2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 6,  7,  8,  9,  10, 11,
            12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23});
}

TEST_F(TileOpTest, TestTileWithDataFormat) {
  TestTileWithDataFormat(
      {2, 2, 2, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
      {1, 1, 1, 2}, {2, 2, 2, 6},
      {0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  7,  8,  6,
       7,  8,  9,  10, 11, 9,  10, 11, 12, 13, 14, 12, 13, 14, 15, 16,
       17, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22, 23});
  TestTileWithDataFormat(
      {2, 2, 2, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
      {1, 2, 2, 1}, {2, 4, 4, 3},
      {0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
       10, 11, 6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  4,  5,  0,  1,
       2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 6,  7,  8,  9,  10, 11,
       12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 12, 13,
       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
