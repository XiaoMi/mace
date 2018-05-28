// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/resize_bilinear.h"

namespace mace {
namespace ops {
namespace test {

class ResizeBilinearTest : public OpsTestBase {};

TEST_F(ResizeBilinearTest, CPUResizeBilinearWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 6, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBilinearTest, ResizeBilinearWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("align_corners", 1)
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 9, 10, 11});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void TestRandomResizeBilinear() {
  testing::internal::LogToStderr();
  static unsigned int seed = time(NULL);
  for (int round = 0; round < 10; ++round) {
    int batch = 1 + rand_r(&seed) % 5;
    int channels = 1 + rand_r(&seed) % 100;
    int height = 1 + rand_r(&seed) % 100;
    int width = 1 + rand_r(&seed) % 100;
    int in_height = 1 + rand_r(&seed) % 100;
    int in_width = 1 + rand_r(&seed) % 100;
    int align_corners = rand_r(&seed) % 1;

    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, in_height, in_width, channels});
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);

    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());
    // Run on CPU
    net.RunOp(DeviceType::CPU);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    if (D == DeviceType::GPU) {
      BufferToImage<D, float>(&net, "Input", "InputImage",
                              kernels::BufferType::IN_OUT_CHANNEL);

      OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
          .Input("InputImage")
          .Output("OutputImage")
          .AddIntArg("align_corners", align_corners)
          .AddIntsArg("size", {height, width})
          .Finalize(net.NewOperatorDef());
      // Run
      net.RunOp(D);

      ImageToBuffer<D, float>(&net, "OutputImage", "DeviceOutput",
                              kernels::BufferType::IN_OUT_CHANNEL);
    }
    // Check
    ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 1e-5,
                            1e-6);
  }
}
}  // namespace

TEST_F(ResizeBilinearTest, OPENCLRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::GPU>();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
