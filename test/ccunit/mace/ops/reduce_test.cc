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

#include "mace/ops/reduce.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ReduceOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const std::vector<index_t> &input_shape,
            const std::vector<float> &input,
            const std::vector<int> &axis,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output,
            ReduceType type,
            const bool keepdims = true) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("Reduce", "ReduceTest")
        .Input("InputNCHW")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", keepdims ? 1 : 0)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else {
    OpDefBuilder("Reduce", "ReduceTest")
        .Input("Input")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", keepdims ? 1 : 0)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }
  auto expected = net.CreateTensor<float>(output_shape, output);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5, 1e-3);
}

template <DeviceType D>
void Simple3D(const std::vector<index_t> &input_shape,
              const std::vector<float> &input,
              const std::vector<int> &axis,
              const std::vector<index_t> &output_shape,
              const std::vector<float> &output,
              ReduceType type,
              const bool keepdims = true) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input);

  OpDefBuilder("Reduce", "ReduceTest")
      .Input("Input")
      .AddIntsArg("axis", axis)
      .AddIntArg("keepdims", keepdims ? 1 : 0)
      .AddIntArg("reduce_type", type)
      .AddIntArg("has_data_format", 1)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  auto expected = net.CreateTensor<float>(output_shape, output);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5, 1e-3);
}

template <DeviceType D>
void SimpleMean12Test() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {1, 2},
            {2, 1, 1, 4},
            {10, 11, 12, 13,
             10, 11, 12, 13}, ReduceType::MEAN);
}

template <DeviceType D>
void SimpleMin12Test() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {1, 2},
            {2, 1, 1, 4},
            {0, 1, 2, 3,
             0, 1, 2, 3}, ReduceType::MIN);
}

template <DeviceType D>
void SimpleMax12Test() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {1, 2},
            {2, 1, 1, 4},
            {20, 21, 22, 23,
             20, 21, 22, 23}, ReduceType::MAX);
}

template <DeviceType D>
void SimpleMean1Axis() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {1},
            {2, 1, 3, 4},
            {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
             6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, ReduceType::MEAN);
}

template <DeviceType D>
void SimpleMin1Axis() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23,
             0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {1},
            {2, 1, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, ReduceType::MIN);
}

template <DeviceType D>
void SimpleMax1Axis() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23,
             0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {1},
            {2, 1, 3, 4},
            {12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23}, ReduceType::MAX);
}

template <DeviceType D>
void Simple2Axis() {
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 1},
            {1, 1, 3, 4},
            {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, ReduceType::MEAN);
  Simple3D<D>({2, 3, 4},
              {0, 1, 2, 3,
               4, 5, 6, 7,
               8, 9, 10, 11,
               12, 13, 14, 15,
               16, 17, 18, 19,
               20, 21, 22, 23},
              {1, 2},
              {2, 1, 1},
              {5.5, 17.5}, ReduceType::MEAN);
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 2},
            {1, 2, 1, 4},
            {4, 5, 6, 7, 16, 17, 18, 19}, ReduceType::MEAN);
}

template <DeviceType D>
void Simple3Axis() {
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {1, 2, 3},
            {1, 1, 1, 1},
            {11.5}, ReduceType::MEAN);
}

}  // namespace

TEST_F(ReduceOpTest, CPUSimple12) {
  SimpleMean12Test<DeviceType::CPU>();
  SimpleMin12Test<DeviceType::CPU>();
  SimpleMax12Test<DeviceType::CPU>();
}

TEST_F(ReduceOpTest, GPUSimple12) {
  SimpleMean12Test<DeviceType::GPU>();
  SimpleMin12Test<DeviceType::GPU>();
  SimpleMax12Test<DeviceType::GPU>();
}

TEST_F(ReduceOpTest, CPUSimple1Axis) {
  SimpleMean1Axis<DeviceType::CPU>();
  SimpleMin1Axis<DeviceType::CPU>();
  SimpleMax1Axis<DeviceType::CPU>();
}

TEST_F(ReduceOpTest, CPUSimple2Axis) {
  Simple2Axis<DeviceType::CPU>();
}

TEST_F(ReduceOpTest, CPUSimple3Axis) {
  Simple3Axis<DeviceType::CPU>();
}

TEST_F(ReduceOpTest, CPUSimpleReduceDims) {
  Simple3D<CPU>({2, 3, 4},
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
              {0, 1},
              {4},
              {10, 11, 12, 13}, ReduceType::MEAN,
              false);
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const std::vector<index_t> &input_shape,
                const std::vector<int> &axis) {
  testing::internal::LogToStderr();
  srand(time(NULL));
  auto func = [&](ReduceType type) {
    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<D, float>("Input", input_shape);

    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("Reduce", "ReduceTest")
        .Input("InputNCHW")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", 1)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp();
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
    OpDefBuilder("Reduce", "ReduceTest")
        .Input("Input")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", 1)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
        .Output("OPENCLOutput")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    if (DataTypeToEnum<T>::value == DT_FLOAT) {
      ExpectTensorNear<float>(*net.GetTensor("Output"),
                              *net.GetOutput("OPENCLOutput"), 1e-5, 1e-4);
    } else {
      ExpectTensorNear<float>(*net.GetTensor("Output"),
                              *net.GetOutput("OPENCLOutput"), 1e-2, 1e-2);
    }
  };

  for (ReduceType type : {MEAN, MIN, MAX, PROD}) {
    func(type);
  }
}
}  // namespace

TEST_F(ReduceOpTest, GPURandomFloat) {
  RandomTest<DeviceType::GPU, float>({4, 64, 64, 3}, {1, 2});
  RandomTest<DeviceType::GPU, float>({8, 128, 128, 64}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 480, 640, 32}, {1, 2});
  RandomTest<DeviceType::GPU, float>({8, 117, 87, 33}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 511, 561, 11}, {1, 2});
}

TEST_F(ReduceOpTest, GPURandomHalf) {
  RandomTest<DeviceType::GPU, half>({4, 64, 64, 3}, {1, 2});
  RandomTest<DeviceType::GPU, half>({8, 128, 128, 64}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 480, 640, 32}, {1, 2});
  RandomTest<DeviceType::GPU, half>({8, 117, 87, 33}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 511, 561, 11}, {1, 2});
}

namespace {

void TestQuant(const std::vector<index_t> &input_shape,
               const std::vector<int> &axis) {
  auto func = [&](ReduceType type) {
    OpsTestNet net;
    net.AddRandomInput<CPU, float>(
        "Input", input_shape, false, false);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    net.AddRandomInput<DeviceType::CPU, float>(
        "OutputNCHW", input_shape, false, true, true);

    OpDefBuilder("Reduce", "ReduceTest")
        .Input("InputNCHW")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", 1)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
        .Output("OutputNCHW")
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

    net.AddRandomInput<DeviceType::CPU, uint8_t>("QuantizedOutput",
                                                 input_shape);
    OpDefBuilder("Reduce", "ReduceTest")
        .Input("QuantizedInput")
        .Output("QuantizedOutput")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", 1)
        .AddIntArg("reduce_type", type)
        .AddIntArg("has_data_format", 1)
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
  };

  for (ReduceType type : {MEAN, MIN, MAX}) {
    func(type);
  }
}
}  // namespace

TEST_F(ReduceOpTest, Quant) {
  // reduce 1, first axis
  TestQuant({1, 1, 3, 4}, {2, 3});
  // reduce 2, first axis
  TestQuant({1, 4, 4, 320}, {1, 2});
  // reduce 2, not first axis
  TestQuant({16, 320, 4, 4}, {2, 3});
  // reduce 3, first axis
  TestQuant({1, 4, 323, 4}, {1, 3});
  // reduce 3, not first axis
  TestQuant({15, 117, 15, 32}, {2});
  // reduce 4, first axis
  TestQuant({4, 323, 4, 4}, {0, 2});
  // reduce 4, not first axis
  TestQuant({32, 4, 323, 16}, {1, 3});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
