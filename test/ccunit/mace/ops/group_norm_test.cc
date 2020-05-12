// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/core/types.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class GroupNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestGroupNorm(const std::vector<index_t> &input_shape,
                const std::vector<T> &input,
                int group_num,
                const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<D, T>(MakeString("Input"), input_shape, input);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  }
  OpDefBuilder("GroupNorm", "GroupNormTest")
      .Input(D == DeviceType::CPU ? "InputNCHW" : "Input")
      .AddIntArg("group_num", group_num)
      .Output(D == DeviceType::CPU ? "OutputNCHW" : "Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }

  net.AddInputFromArray<D, T>("ExpectedOutput", input_shape, output);
  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                        *net.GetOutput("Output"), 1e-2, 1e-2);
  } else {
    ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                        *net.GetOutput("Output"), 1e-3);
  }
}
}  // namespace

TEST_F(GroupNormOpTest, SimpleTestCPU) {
  TestGroupNorm<DeviceType::CPU, float>(
    {1, 1, 2, 64},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     1, 2, 3, 4, 5, 6, 7, 8},
    16,
    {-1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
     -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
     1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
     -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
     1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
     -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
     1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
     -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
     1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
     -1.09104, -0.654625, -0.218208, 0.80467, 0.928465, 1.05226,
     1.17606, -1.52746, -1.09104, -0.654625, -0.218208, 0.218208,
     0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
     1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
     0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
     1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
     0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
     1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
     0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
     1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
     0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
     1.52746, -1.17606, -1.05226, -0.928465, -0.80467, 0.218208,
     0.654625, 1.09104, 1.52746});
}


TEST_F(GroupNormOpTest, SimpleTestOpenCL) {
  TestGroupNorm<DeviceType::GPU, float>(
  {1, 1, 2, 64},
  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
   3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
   5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
   9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
   3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
   5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
   9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
   1, 2, 3, 4, 5, 6, 7, 8},
  16,
  {-1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
   -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
   1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
   -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
   1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
   -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
   1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
   -1.09104, -0.654625, -0.218208, 0.468507, 0.780844, 1.09318,
   1.40552, -1.52746, -1.09104, -0.654625, -0.218208, -1.52746,
   -1.09104, -0.654625, -0.218208, 0.80467, 0.928465, 1.05226,
   1.17606, -1.52746, -1.09104, -0.654625, -0.218208, 0.218208,
   0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
   1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
   0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
   1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
   0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
   1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
   0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
   1.52746, -1.40552, -1.09318, -0.780844, -0.468507, 0.218208,
   0.654625, 1.09104, 1.52746, 0.218208, 0.654625, 1.09104,
   1.52746, -1.17606, -1.05226, -0.928465, -0.80467, 0.218208,
   0.654625, 1.09104, 1.52746});
}


TEST_F(GroupNormOpTest, SimpleRandomOPENCL) {
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 5;
  index_t group = 4 + 4 * (rand_r(&seed) % 3);
  index_t group_num = 2 + rand_r(&seed) % 16;
  index_t channels = group * group_num;
  index_t height = 64;
  index_t width = 64;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("GroupNorm", "GroupNormTest")
      .Input("InputNCHW")
      .AddFloatArg("epsilon", 1e-3)
      .Output("OutputNCHW")
      .AddIntArg("group_num", group_num)
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("GroupNorm", "GroupNormTest")
      .Input("Input")
      .AddFloatArg("epsilon", 1e-3)
      .Output("Output")
      .AddIntArg("group_num", group_num)
      .Finalize(net.NewOperatorDef());

  net.Setup(DeviceType::GPU);

  // Tuning
  setenv("MACE_TUNING", "1", 1);
  net.Run();
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.Run();

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-5, 1e-4);
}


TEST_F(GroupNormOpTest, SimpleRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 5;
  index_t group = 4 + 4 * (rand_r(&seed) % 16);
  index_t group_num = 2 + rand_r(&seed) % 16;
  index_t channels = group * group_num;
  index_t height = 64;
  index_t width = 64;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("GroupNorm", "GroupNormTest")
      .Input("InputNCHW")
      .AddFloatArg("epsilon", 1e-3)
      .Output("OutputNCHW")
      .AddIntArg("group_num", group_num)
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("GroupNorm", "GroupNormTest")
      .Input("Input")
      .AddFloatArg("epsilon", 1e-3)
      .Output("Output")
      .AddIntArg("group_num", group_num)
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  net.Setup(DeviceType::GPU);

  // Tuning
  setenv("MACE_TUNING", "1", 1);
  net.Run();
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.Run();

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-1, 1e-2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
