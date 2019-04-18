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

#include "mace/ops/pooling.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class PoolingOpTest : public OpsTestBase {};

TEST_F(PoolingOpTest, MAX_VALID) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected =
      net.CreateTensor<float>({1, 2, 2, 2}, {5, 21, 7, 23, 13, 29, 15, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(PoolingOpTest, MAX_SAME) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 3, 1},
                                                {0, 1, 2, 3, 4, 5, 6, 7, 8});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>({1, 2, 2, 1}, {4, 5, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(PoolingOpTest, MAX_VALID_DILATION) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 4, 4, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {2, 2})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>({1, 2, 2, 1}, {10, 11, 14, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(PoolingOpTest, MAX_k2x2s2x2) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 2, 9, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("pooling_type", PoolingType::MAX)
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
  auto expected = net.CreateTensor<float>({1, 1, 5, 1}, {10, 12, 14, 16, 17});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void SimpleMaxPooling3S2() {
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
    OpDefBuilder("Pooling", "PoolingTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("pooling_type", PoolingType::MAX)
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .AddIntArg("pooling_type", PoolingType::MAX)
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
  }

  // Check
  auto expected = net.CreateTensor<float>({1, 1, 4, 1}, {20, 22, 24, 26});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(PoolingOpTest, CPUSimpleMaxPooling3S2) { SimpleMaxPooling3S2<CPU>(); }

TEST_F(PoolingOpTest, OPENCLSimpleMaxPooling3S2) { SimpleMaxPooling3S2<GPU>(); }

namespace {
template <DeviceType D, typename T>
void MaxPooling3S2(const std::vector<index_t> &input_shape,
                   const std::vector<int> &strides,
                   Padding padding) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("pooling_type", PoolingType::MAX)
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

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
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

TEST_F(PoolingOpTest, OPENCLAlignedMaxPooling3S2) {
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {1, 2}, Padding::VALID);
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
  MaxPooling3S2<GPU, float>({3, 64, 32, 32}, {2, 1}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedMaxPooling3S2) {
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {1, 2}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
  MaxPooling3S2<GPU, half>({3, 64, 32, 32}, {2, 1}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnalignedMaxPooling3S2) {
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {1, 1}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {2, 2}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {1, 2}, Padding::VALID);
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {1, 1}, Padding::SAME);
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {2, 2}, Padding::SAME);
  MaxPooling3S2<GPU, half>({3, 41, 43, 47}, {2, 1}, Padding::SAME);
}

TEST_F(PoolingOpTest, AVG_VALID) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::AVG)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 2, 2, 2}, {2.5, 18.5, 4.5, 20.5, 10.5, 26.5, 12.5, 28.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void SimpleAvgPoolingTest() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 8, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  // Check
  auto expected = net.CreateTensor<float>({1, 1, 4, 1}, {4.5, 6.5, 8.5, 10.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(PoolingOpTest, OPENCLSimpleAvgPooling) { SimpleAvgPoolingTest<GPU>(); }

namespace {
template <DeviceType D, typename T>
void AvgPoolingTest(const std::vector<index_t> &shape,
                    const std::vector<int> &kernels,
                    const std::vector<int> &strides,
                    Padding padding) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", shape);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", kernels)
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

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", kernels)
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-3,
                            1e-3);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  }
}
}  // namespace

TEST_F(PoolingOpTest, OPENCLAlignedAvgPooling) {
  AvgPoolingTest<GPU, float>({3, 15, 15, 128}, {4, 4}, {4, 4}, Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 15, 15, 128}, {3, 4}, {1, 2}, Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 15, 15, 128}, {4, 4}, {4, 4}, Padding::SAME);
  AvgPoolingTest<GPU, float>({3, 15, 15, 128}, {3, 4}, {1, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedAvgPooling) {
  AvgPoolingTest<GPU, half>({3, 15, 15, 128}, {4, 4}, {4, 4}, Padding::VALID);
  AvgPoolingTest<GPU, half>({3, 15, 15, 128}, {3, 4}, {1, 2}, Padding::VALID);
  AvgPoolingTest<GPU, half>({3, 15, 15, 128}, {4, 4}, {4, 4}, Padding::SAME);
  AvgPoolingTest<GPU, half>({3, 15, 15, 128}, {3, 4}, {1, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<GPU, float>({3, 64, 64, 128}, {16, 16}, {16, 16},
                             Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 64, 64, 128}, {12, 16}, {12, 8},
                             Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 64, 64, 128}, {16, 16}, {16, 16},
                             Padding::SAME);
  AvgPoolingTest<GPU, float>({3, 64, 64, 128}, {8, 16}, {8, 16},
                             Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<GPU, half>({3, 64, 64, 128}, {16, 16}, {16, 16},
                            Padding::VALID);
  AvgPoolingTest<GPU, half>({3, 64, 64, 128}, {16, 16}, {16, 16},
                            Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnAlignedAvgPooling) {
  AvgPoolingTest<GPU, float>({3, 31, 37, 128}, {2, 2}, {2, 2}, Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 31, 37, 128}, {2, 2}, {2, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<GPU, float>({3, 31, 37, 128}, {8, 8}, {8, 8}, Padding::VALID);
  AvgPoolingTest<GPU, float>({3, 31, 37, 128}, {8, 8}, {8, 8}, Padding::SAME);
}

TEST_F(PoolingOpTest, QUANT_MAX_VALID) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, uint8_t>(
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  // Check
  auto expected =
      net.CreateTensor<uint8_t>({1, 2, 2, 2}, {5, 21, 7, 23, 13, 29, 15, 31});

  ExpectTensorNear<uint8_t>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(PoolingOpTest, QUANT_MAX_SAME) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, uint8_t>("Input", {1, 3, 3, 1},
                                                  {0, 1, 2, 3, 4, 5, 6, 7, 8});

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  // Check
  auto expected = net.CreateTensor<uint8_t>({1, 2, 2, 1}, {4, 5, 7, 8});

  ExpectTensorNear<uint8_t>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(PoolingOpTest, QUANT_AVG_VALID) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, uint8_t>(
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  // Check
  auto expected = net.CreateTensor<uint8_t>(
      {1, 2, 2, 2}, {3, 19, 5, 21, 11, 27, 13, 29});

  ExpectTensorNear<uint8_t>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {

void TestQuant(const index_t batch,
               const index_t in_height,
               const index_t in_width,
               const index_t channels,
               const std::vector<int> &kernels,
               const std::vector<int> &strides,
               enum Padding padding_type,
               PoolingType pooling) {
  OpsTestNet net;
  std::vector<index_t> input_shape{batch, in_height, in_width, channels};
  net.AddRandomInput<CPU, float>(
      "Input", input_shape, false, false);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  net.AddRandomInput<DeviceType::CPU, float>(
      "OutputNCHW", input_shape, false, true, true);
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("pooling_type", pooling)
      .AddIntsArg("kernels", kernels)
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding_type)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", DT_FLOAT)
      .Finalize(net.NewOperatorDef());

  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  net.AddRandomInput<DeviceType::CPU, uint8_t>("QuantizedOutput", input_shape);
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("QuantizedInput")
      .Output("QuantizedOutput")
      .AddIntsArg("kernels", kernels)
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding_type)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", pooling)
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}
}  // namespace

TEST_F(PoolingOpTest, Quant) {
  TestQuant(1, 7, 7, 1024, {7, 7}, {1, 1}, Padding::VALID, PoolingType::AVG);
  TestQuant(1, 3, 3, 2, {3, 3}, {1, 1}, Padding::SAME, PoolingType::AVG);
  TestQuant(1, 3, 3, 2, {2, 3}, {1, 2}, Padding::SAME, PoolingType::AVG);
  TestQuant(1, 7, 7, 1024, {7, 7}, {1, 1}, Padding::VALID, PoolingType::MAX);
  TestQuant(1, 7, 7, 1024, {7, 7}, {1, 1}, Padding::SAME, PoolingType::MAX);
  TestQuant(1, 7, 7, 1024, {6, 7}, {4, 5}, Padding::SAME, PoolingType::MAX);
  TestQuant(1, 7, 7, 2048, {7, 7}, {1, 1}, Padding::SAME, PoolingType::AVG);
  TestQuant(3, 15, 15, 128, {4, 4}, {4, 4}, Padding::VALID, PoolingType::AVG);
  TestQuant(3, 15, 15, 128, {4, 4}, {4, 4}, Padding::VALID, PoolingType::MAX);
  TestQuant(3, 31, 37, 128, {2, 2}, {2, 2}, Padding::VALID, PoolingType::AVG);
  TestQuant(3, 31, 37, 128, {2, 2}, {2, 2}, Padding::VALID, PoolingType::MAX);
}
}  // namespace test
}  // namespace ops
}  // namespace mace
