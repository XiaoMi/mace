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

#include <fstream>

#include <vector>
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D>
void RunSpaceToDepth(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const int block_size,
                     const std::vector<index_t> &expected_shape,
                     const std::vector<float> &expected_data) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);
  // Construct graph
  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("block_size", block_size)
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  } else {
    OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
        .Input("Input")
        .Output("Output")
        .AddIntArg("block_size", block_size)
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  auto expected = net.CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

class SpaceToDepthOpTest : public OpsTestBase {};

TEST_F(SpaceToDepthOpTest, CPUInputDepthLess4) {
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 3, 6, 1},
      {0, 1, 2, 9, 10, 11,
       3, 4, 5, 12, 13, 14,
       6, 7, 8, 15, 16, 17},
      3,
      {1, 1, 2, 9},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 3, 6, 2},
      {0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23,
       6, 7, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29,
       12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35},
      3,
      {1, 1, 2, 18},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 2, 4, 3},
      {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17,
       6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23},
      2,
      {1, 1, 2, 12},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 3, 3, 3},
      {0, 1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26},
      3,
      {1, 1, 1, 27},
      {0, 1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26});
}

TEST_F(SpaceToDepthOpTest, CPUInputDepth4) {
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 2, 4, 4},
      {0, 1, 2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
       8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31},
      2, {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  RunSpaceToDepth<DeviceType::CPU>(
      {1, 2, 2, 4},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 2, {1, 1, 1, 16},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(SpaceToDepthOpTest, OPENCLInputDepth1) {
  RunSpaceToDepth<DeviceType::GPU>(
      {1, 3, 6, 1},
      {0, 1, 2, 9, 10, 11,
       3, 4, 5, 12, 13, 14,
       6, 7, 8, 15, 16, 17},
      3, {1, 1, 2, 9},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
}

TEST_F(SpaceToDepthOpTest, OPENCLInputDepth2) {
  RunSpaceToDepth<DeviceType::GPU>(
      {1, 3, 6, 2},
      {0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23,
       6, 7, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29,
      12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35},
      3, {1, 1, 2, 18},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
}

TEST_F(SpaceToDepthOpTest, OPENCLInputDepth3) {
  RunSpaceToDepth<DeviceType::GPU>(
      {1, 2, 4, 3},
      {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17,
       6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23},
      2, {1, 1, 2, 12},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  RunSpaceToDepth<DeviceType::GPU>(
      {1, 3, 3, 3},
      {0, 1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26},
      3, {1, 1, 1, 27},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,
       9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26});
}

TEST_F(SpaceToDepthOpTest, OPENCLInputDepth4) {
  RunSpaceToDepth<DeviceType::GPU>(
      {1, 2, 4, 4},
      {0, 1, 2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
       8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31},
      2, {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const int block_size,
                const std::vector<index_t> &shape) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", shape);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
      .Input("InputNCHW")
      .AddIntArg("block_size", block_size)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
      .Input("Input")
      .AddIntArg("block_size", block_size)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("OPENCLOutput")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-5);
  } else {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-3, 1e-4);
  }
}
}  // namespace

TEST_F(SpaceToDepthOpTest, OPENCLRandomFloatDepth1) {
  RandomTest<DeviceType::GPU, float>(2, {1, 384, 384, 1});
  RandomTest<DeviceType::GPU, float>(3, {1, 333, 333, 1});
  RandomTest<DeviceType::GPU, float>(5, {1, 100, 100, 1});
  RandomTest<DeviceType::GPU, float>(7, {1, 98, 98, 1});
}

TEST_F(SpaceToDepthOpTest, OPENCLRandomFloatDepth2) {
  RandomTest<DeviceType::GPU, float>(2, {1, 384, 384, 2});
  RandomTest<DeviceType::GPU, float>(3, {1, 333, 333, 2});
  RandomTest<DeviceType::GPU, float>(5, {1, 100, 100, 2});
  RandomTest<DeviceType::GPU, float>(7, {1, 98, 98, 2});
}

TEST_F(SpaceToDepthOpTest, OPENCLRandomFloatDepth3) {
  RandomTest<DeviceType::GPU, float>(2, {1, 384, 384, 3});
  RandomTest<DeviceType::GPU, float>(3, {1, 333, 333, 3});
  RandomTest<DeviceType::GPU, float>(5, {1, 100, 100, 3});
  RandomTest<DeviceType::GPU, float>(7, {1, 98, 98, 3});
}

TEST_F(SpaceToDepthOpTest, OPENCLRandomFloat) {
  RandomTest<DeviceType::GPU, float>(2, {1, 384, 384, 4});
  RandomTest<DeviceType::GPU, float>(3, {1, 333, 333, 16});
  RandomTest<DeviceType::GPU, float>(5, {1, 100, 100, 32});
  RandomTest<DeviceType::GPU, float>(7, {1, 98, 98, 64});
}

TEST_F(SpaceToDepthOpTest, OPENCLRandomHalf) {
  RandomTest<DeviceType::GPU, half>(2, {1, 384, 384, 1});
  RandomTest<DeviceType::GPU, half>(3, {1, 333, 333, 2});
  RandomTest<DeviceType::GPU, half>(5, {1, 100, 100, 3});
  RandomTest<DeviceType::GPU, half>(7, {1, 98, 98, 4});
  RandomTest<DeviceType::GPU, half>(2, {1, 384, 384, 32});
}

TEST_F(SpaceToDepthOpTest, OPENCLBatchRandomHalf) {
  RandomTest<DeviceType::GPU, half>(2, {2, 384, 384, 1});
  RandomTest<DeviceType::GPU, half>(3, {3, 333, 333, 2});
  RandomTest<DeviceType::GPU, half>(5, {2, 100, 100, 3});
  RandomTest<DeviceType::GPU, half>(7, {3, 98, 98, 4});
  RandomTest<DeviceType::GPU, half>(2, {2, 384, 384, 32});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
