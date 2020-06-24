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

#include <vector>

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ExtractImagePatchesOpTest : public OpsTestBase {};

TEST_F(ExtractImagePatchesOpTest, VALID) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected =
      net.CreateTensor<float>({1, 2, 2, 8},
          {0, 16, 1, 17, 4, 20, 5, 21, 2, 18, 3, 19, 6, 22, 7, 23,
           8, 24, 9, 25, 12, 28, 13, 29, 10, 26, 11, 27, 14, 30, 15, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ExtractImagePatchesOpTest, SAME) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 3, 1},
                                                {0, 1, 2, 3, 4, 5, 6, 7, 8});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 2, 2, 4}, {0, 1, 3, 4, 2, 0, 5, 0, 6, 7, 0, 0, 8, 0, 0, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ExtractImagePatchesOpTest, VALID_DILATION) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 4, 4, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {2, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 2, 2, 4}, {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ExtractImagePatchesOpTest, k2x2s2x2) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 2, 9, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 1, 5, 4},
      {0, 1, 9, 10, 2, 3, 11, 12, 4, 5, 13, 14, 6, 7, 15, 16, 8, 0, 17, 0});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void SimpleExtractImagePatches3S2() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 9, 1},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    // Run
    OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
  }

  // Check
  auto expected = net.CreateTensor<float>({1, 1, 4, 9},
                                          {0, 1, 2, 9, 10, 11, 18, 19, 20,
                                           2, 3, 4, 11, 12, 13, 20, 21, 22,
                                           4, 5, 6, 13, 14, 15, 22, 23, 24,
                                           6, 7, 8, 15, 16, 17, 24, 25, 26});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ExtractImagePatchesOpTest, CPUSimpleExtractImagePatches3S2) {
  SimpleExtractImagePatches3S2<CPU>();
}

namespace {
template <DeviceType D, typename T>
void ExtractImagePatches3S2(const std::vector<index_t> &input_shape,
                            const std::vector<int> &strides,
                            Padding padding) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // run on cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("ExtractImagePatches", "ExtractImagePatchesTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-3,
                            1e-4);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  }
}
}  // namespace

TEST_F(ExtractImagePatchesOpTest, OPENCLAlignedExtractImagePatches3S2) {
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {1, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
  ExtractImagePatches3S2<GPU, float>({3, 64, 32, 32}, {2, 1}, Padding::SAME);
  ExtractImagePatches3S2<GPU, float>({3, 63, 31, 32}, {2, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, float>({3, 65, 27, 32}, {2, 1}, Padding::SAME);
}

TEST_F(ExtractImagePatchesOpTest, OPENCLHalfAlignedExtractImagePatches3S2) {
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {1, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
  ExtractImagePatches3S2<GPU, half>({3, 64, 32, 32}, {2, 1}, Padding::SAME);
  ExtractImagePatches3S2<GPU, half>({3, 63, 31, 32}, {2, 2}, Padding::VALID);
  ExtractImagePatches3S2<GPU, half>({3, 65, 27, 32}, {2, 1}, Padding::SAME);
}


}  // namespace test
}  // namespace ops
}  // namespace mace
