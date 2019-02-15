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
#include "mace/ops/pad.h"
#include <functional>
#include <string>
#include <vector>

namespace mace {
namespace ops {
namespace test {

class PadTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<D, float>("Input", {1, 2, 3, 1}, 2);
  if (D == DeviceType::GPU) {
    OpDefBuilder("Pad", "PadTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("paddings", {0, 0, 1, 2, 1, 2, 0, 0})
        .AddFloatArg("constant_value", 1.0)
        .AddIntArg("data_format", DataFormat::NHWC)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                    NCHW);
    OpDefBuilder("Pad", "PadTest")
        .Input("TInput")
        .Output("TOutput")
        .AddIntsArg("paddings", {0, 0, 1, 2, 1, 2, 0, 0})
        .AddFloatArg("constant_value", 1.0)
        .AddIntArg("data_format", DataFormat::NHWC)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                    NHWC);
  }

  auto output = net.GetTensor("Output");

  auto expected = net.CreateTensor<float>(
      {1, 5, 6, 1}, {
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2,   2,   2,
                        1.0, 1.0, 1.0, 2,   2,   2,   1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}
}  // namespace

TEST_F(PadTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

TEST_F(PadTest, SimpleGPU) { Simple<DeviceType::GPU>(); }

TEST_F(PadTest, ComplexCPU) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<DeviceType::CPU, float>("Input", {1, 1, 1, 2}, 2);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                  NCHW);
  OpDefBuilder("Pad", "PadTest")
      .Input("TInput")
      .Output("TOutput")
      .AddIntsArg("paddings", {0, 0, 1, 1, 1, 1, 1, 1})
      .AddFloatArg("constant_value", 1.0)
      .AddIntArg("data_format", DataFormat::NHWC)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);

  auto output = net.GetTensor("Output");

  auto expected = net.CreateTensor<float>(
      {1, 3, 3, 4},
      {
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}

namespace {
template <typename T>
void Complex(const std::vector<index_t> &input_shape,
             const std::vector<int> &paddings, const int pad_type) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                  NCHW);
  OpDefBuilder("Pad", "PadTest")
      .Input("TInput")
      .Output("TOutput")
      .AddIntsArg("paddings", paddings)
      .AddIntArg("pad_type", pad_type)
      .AddFloatArg("constant_value", 1.0)
      .AddIntArg("data_format", DataFormat::NHWC)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);

  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("Pad", "PadTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("paddings", paddings)
      .AddIntArg("pad_type", pad_type)
      .AddFloatArg("constant_value", 1.0)
      .AddIntArg("data_format", DataFormat::NHWC)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  auto output = net.GetTensor("Output");

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *output, 1e-2, 1e-2);
  } else {
    ExpectTensorNear<float>(*expected, *output, 1e-5);
  }
}
}  // namespace

TEST_F(PadTest, ComplexFloat) {
  for (int i = PadType::CONSTANT; i <= PadType::SYMMETRIC; i++) {
    Complex<float>({1, 32, 32, 4}, {0, 0, 2, 2, 1, 1, 0, 0}, i);
    Complex<float>({1, 31, 37, 16}, {0, 0, 2, 0, 1, 0, 0, 0}, i);
    Complex<float>({1, 128, 128, 32}, {0, 0, 0, 1, 0, 2, 0, 0}, i);
  }
}

TEST_F(PadTest, ComplexHalf) {
  for (int i = PadType::CONSTANT; i <= PadType::SYMMETRIC; i++) {
    Complex<half>({1, 32, 32, 4}, {0, 0, 2, 2, 1, 1, 0, 0}, i);
    Complex<half>({1, 31, 37, 16}, {0, 0, 2, 0, 1, 0, 0, 0}, i);
    Complex<half>({1, 128, 128, 32}, {0, 0, 0, 1, 0, 2, 0, 0}, i);
  }
}

namespace {
template <DeviceType D, typename T>
void Result(const std::vector<index_t> &input_shape,
            const std::vector<float> &input_data,
            const std::vector<index_t> &expected_shape,
            const std::vector<float> &expected_data,
            const std::vector<int> &paddings,
            const PadType pad_type) {
  // Construct graph
  OpsTestNet net;
  std::string input("Input");
  std::string t_input(input);
  std::string output("Output");
  std::string t_output(output);

  // Add input data
  net.AddInputFromArray<D, float>(input, input_shape, input_data);

  if (D == DeviceType::CPU) {
    t_input = "TInput";
    t_output = "TOutput";
    net.TransformDataFormat<DeviceType::CPU, T>(input, NHWC, t_input, NCHW);
  }

  OpDefBuilder("Pad", "PadTest")
  .Input(t_input)
  .Output(t_output)
  .AddIntsArg("paddings", paddings)
  .AddIntArg("pad_type", static_cast<int>(pad_type))
  .AddIntArg("data_format", DataFormat::NHWC)
  .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, T>(t_output, NCHW, output, NHWC);
  }

  auto actual = net.GetTensor(output.c_str());
  auto expected = net.CreateTensor<float>(expected_shape, expected_data);

  ExpectTensorNear<float>(*expected, *actual, 1e-5);
}
}  // namespace

TEST_F(PadTest, ReflectCPU) {
  std::vector<index_t> input_shape{2, 2, 2, 2};
  int size = std::accumulate(input_shape.begin(), input_shape.end(),
                             1, std::multiplies<index_t>());
  std::vector<float> input_data;
  std::vector<index_t> expected_shape{4, 4, 4, 4};
  std::vector<float> expected_data{
          16, 15, 16, 15,
          14, 13, 14, 13,
          16, 15, 16, 15,
          14, 13, 14, 13,

          12, 11, 12, 11,
          10,  9, 10,  9,
          12, 11, 12, 11,
          10,  9, 10,  9,

          16, 15, 16, 15,
          14, 13, 14, 13,
          16, 15, 16, 15,
          14, 13, 14, 13,

          12, 11, 12, 11,
          10,  9, 10,  9,
          12, 11, 12, 11,
          10,  9, 10,  9,


           8,  7,  8,  7,
           6,  5,  6,  5,
           8,  7,  8,  7,
           6,  5,  6,  5,

           4,  3,  4,  3,
           2,  1,  2,  1,
           4,  3,  4,  3,
           2,  1,  2,  1,

           8,  7,  8,  7,
           6,  5,  6,  5,
           8,  7,  8,  7,
           6,  5,  6,  5,

           4,  3,  4,  3,
           2,  1,  2,  1,
           4,  3,  4,  3,
           2,  1,  2,  1,


          16, 15, 16, 15,
          14, 13, 14, 13,
          16, 15, 16, 15,
          14, 13, 14, 13,

          12, 11, 12, 11,
          10,  9, 10,  9,
          12, 11, 12, 11,
          10,  9, 10,  9,

          16, 15, 16, 15,
          14, 13, 14, 13,
          16, 15, 16, 15,
          14, 13, 14, 13,

          12, 11, 12, 11,
          10,  9, 10,  9,
          12, 11, 12, 11,
          10,  9, 10,  9,


           8,  7,  8,  7,
           6,  5,  6,  5,
           8,  7,  8,  7,
           6,  5,  6,  5,

           4,  3,  4,  3,
           2,  1,  2,  1,
           4,  3,  4,  3,
           2,  1,  2,  1,

           8,  7,  8,  7,
           6,  5,  6,  5,
           8,  7,  8,  7,
           6,  5,  6,  5,

           4,  3,  4,  3,
           2,  1,  2,  1,
           4,  3,  4,  3,
           2,  1,  2,  1,
  };
  const std::vector<int> paddings{1, 1, 1, 1, 1, 1, 1, 1};

  input_data.reserve(size);
  for (int i = 1; i <= size; i++) {
    input_data.push_back(i);
  }

  Result<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
      expected_data, paddings, PadType::REFLECT);
}

TEST_F(PadTest, SymmetricCPU) {
  std::vector<index_t> input_shape{2, 2, 2, 2};
  int size = std::accumulate(input_shape.begin(), input_shape.end(),
                             1, std::multiplies<index_t>());
  std::vector<float> input_data;
  std::vector<index_t> expected_shape{4, 4, 4, 4};
  std::vector<float> expected_data{
           1,  1,  2,  2,
           1,  1,  2,  2,
           3,  3,  4,  4,
           3,  3,  4,  4,

           1,  1,  2,  2,
           1,  1,  2,  2,
           3,  3,  4,  4,
           3,  3,  4,  4,

           5,  5,  6,  6,
           5,  5,  6,  6,
           7,  7,  8,  8,
           7,  7,  8,  8,

           5,  5,  6,  6,
           5,  5,  6,  6,
           7,  7,  8,  8,
           7,  7,  8,  8,


           1,  1,  2,  2,
           1,  1,  2,  2,
           3,  3,  4,  4,
           3,  3,  4,  4,

           1,  1,  2,  2,
           1,  1,  2,  2,
           3,  3,  4,  4,
           3,  3,  4,  4,

           5,  5,  6,  6,
           5,  5,  6,  6,
           7,  7,  8,  8,
           7,  7,  8,  8,

           5,  5,  6,  6,
           5,  5,  6,  6,
           7,  7,  8,  8,
           7,  7,  8,  8,


           9,  9, 10, 10,
           9,  9, 10, 10,
          11, 11, 12, 12,
          11, 11, 12, 12,

           9,  9, 10, 10,
           9,  9, 10, 10,
          11, 11, 12, 12,
          11, 11, 12, 12,

          13, 13, 14, 14,
          13, 13, 14, 14,
          15, 15, 16, 16,
          15, 15, 16, 16,

          13, 13, 14, 14,
          13, 13, 14, 14,
          15, 15, 16, 16,
          15, 15, 16, 16,


           9,  9, 10, 10,
           9,  9, 10, 10,
          11, 11, 12, 12,
          11, 11, 12, 12,

           9,  9, 10, 10,
           9,  9, 10, 10,
          11, 11, 12, 12,
          11, 11, 12, 12,

          13, 13, 14, 14,
          13, 13, 14, 14,
          15, 15, 16, 16,
          15, 15, 16, 16,

          13, 13, 14, 14,
          13, 13, 14, 14,
          15, 15, 16, 16,
          15, 15, 16, 16,
  };
  const std::vector<int> paddings{1, 1, 1, 1, 1, 1, 1, 1};

  input_data.reserve(size);
  for (int i = 1; i <= size; i++) {
    input_data.push_back(i);
  }

  Result<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
      expected_data, paddings, PadType::SYMMETRIC);
}

TEST_F(PadTest, Result) {
  std::vector<index_t> input_shape{1, 3, 4, 1};
  int size = std::accumulate(input_shape.begin(), input_shape.end(),
                             1, std::multiplies<index_t>());
  std::vector<float> input_data;
  std::vector<index_t> expected_shape{1, 6, 7, 1};
  std::vector<float> expected_reflect{
     8,  7,  6, 5,  6,  7,  8,
     4,  3,  2, 1,  2,  3,  4,
     8,  7,  6, 5,  6,  7,  8,
    12, 11, 10, 9, 10, 11, 12,
     8,  7,  6, 5,  6,  7,  8,
     4,  3,  2, 1,  2,  3,  4,
  };
  std::vector<float> expected_symmetric{
     3,  2, 1, 1,  2,  3,  4,
     3,  2, 1, 1,  2,  3,  4,
     7,  6, 5, 5,  6,  7,  8,
    11, 10, 9, 9, 10, 11, 12,
    11, 10, 9, 9, 10, 11, 12,
     7,  6, 5, 5,  6,  7,  8,
  };
  const std::vector<int> paddings{0, 0, 1, 2, 3, 0, 0, 0};

  input_data.reserve(size);
  for (int i = 1; i <= size; i++) {
    input_data.push_back(i);
  }

  Result<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
      expected_reflect, paddings, PadType::REFLECT);
  Result<DeviceType::GPU, float>(input_shape, input_data, expected_shape,
      expected_reflect, paddings, PadType::REFLECT);
  Result<DeviceType::GPU, half>(input_shape, input_data, expected_shape,
      expected_reflect, paddings, PadType::REFLECT);

  Result<DeviceType::CPU, float>(input_shape, input_data, expected_shape,
      expected_symmetric, paddings, PadType::SYMMETRIC);
  Result<DeviceType::GPU, float>(input_shape, input_data, expected_shape,
      expected_symmetric, paddings, PadType::SYMMETRIC);
  Result<DeviceType::GPU, half>(input_shape, input_data, expected_shape,
      expected_symmetric, paddings, PadType::SYMMETRIC);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
