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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ReduceMeanOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const std::vector<index_t> &input_shape,
            const std::vector<float> &input,
            const std::vector<int> &axis,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output,
            const bool keepdims = true) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>("Input", NHWC, "InputNCHW", NCHW);
    OpDefBuilder("ReduceMean", "ReduceMeanTest")
        .Input("InputNCHW")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", keepdims ? 1 : 0)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>("OutputNCHW", NCHW, "Output", NHWC);
  } else {
    OpDefBuilder("ReduceMean", "ReduceMeanTest")
        .Input("Input")
        .AddIntsArg("axis", axis)
        .AddIntArg("keepdims", keepdims ? 1 : 0)
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
              const bool keepdims = true) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input);

  OpDefBuilder("ReduceMean", "ReduceMeanTest")
      .Input("Input")
      .AddIntsArg("axis", axis)
      .AddIntArg("keepdims", keepdims ? 1 : 0)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  auto expected = net.CreateTensor<float>(output_shape, output);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5, 1e-3);
}

template <DeviceType D>
void Simple12Test() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {1, 2},
            {2, 1, 1, 4},
            {10, 11, 12, 13,
             10, 11, 12, 13});
}

template <DeviceType D>
void Simple1Axis() {
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
            {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
             6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {-3},
            {1, 1, 3, 4},
            {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {2},
            {1, 2, 1, 4},
            {4, 5, 6, 7, 16, 17, 18, 19});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {-1},
            {1, 2, 3, 1},
            {1.5, 5.5, 9.5, 13.5, 17.5, 21.5});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {1},
            {1, 1, 3, 3},
            {9, 10, 11, 12, 13, 14, 15, 16, 17});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {-2},
            {1, 3, 1, 3},
            {3, 4, 5, 12, 13, 14, 21, 22, 23});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {3},
            {1, 3, 3, 1},
            {1, 4, 7, 10, 13, 16, 19, 22, 25});
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
            {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 2},
            {1, 2, 1, 4},
            {4, 5, 6, 7, 16, 17, 18, 19});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {1, 3},
            {1, 1, 3, 1},
            {7.5, 11.5, 15.5});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {1, 2},
            {1, 1, 1, 3},
            {12, 13, 14});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {0, 1},
            {1, 1, 3, 3},
            {9, 10, 11, 12, 13, 14, 15, 16, 17});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {2, 3},
            {1, 3, 1, 1},
            {4, 13, 22});
}

template <DeviceType D>
void Simple2Axis3D() {
  Simple3D<D>({2, 3, 4},
              {0, 1, 2, 3,
               4, 5, 6, 7,
               8, 9, 10, 11,
               12, 13, 14, 15,
               16, 17, 18, 19,
               20, 21, 22, 23},
              {0, 1},
              {1, 1, 4},
              {10, 11, 12, 13});
  Simple3D<D>({2, 3, 4},
              {0, 1, 2, 3,
               4, 5, 6, 7,
               8, 9, 10, 11,
               12, 13, 14, 15,
               16, 17, 18, 19,
               20, 21, 22, 23},
              {1, 2},
              {2, 1, 1},
              {5.5, 17.5});
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
            {11.5});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 2, 3},
            {1, 2, 1, 1},
            {5.5, 17.5});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 1, 3},
            {1, 1, 3, 1},
            {7.5, 11.5, 15.5});
  Simple<D>({1, 2, 3, 4},
            {0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22, 23},
            {0, 1, 2},
            {1, 1, 1, 4},
            {10, 11, 12, 13});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {1, 2, 3},
            {1, 1, 1, 1},
            {13});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {0, 2, 3},
            {1, 3, 1, 1},
            {4, 13, 22});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {0, 1, 3},
            {1, 1, 3, 1},
            {10, 13, 16});
  Simple<D>({1, 3, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26},
            {0, 1, 2},
            {1, 1, 1, 3},
            {12, 13, 14});
}

}  // namespace

TEST_F(ReduceMeanOpTest, CPUSimple12) {
  Simple12Test<DeviceType::CPU>();
}

TEST_F(ReduceMeanOpTest, GPUSimple12) {
  Simple12Test<DeviceType::GPU>();
}

TEST_F(ReduceMeanOpTest, CPUSimple1Axis) {
  Simple1Axis<DeviceType::CPU>();
}

TEST_F(ReduceMeanOpTest, CPUSimple2Axis) {
  Simple2Axis<DeviceType::CPU>();
}

TEST_F(ReduceMeanOpTest, CPUSimple2Axis3D) {
  Simple2Axis3D<DeviceType::CPU>();
}

TEST_F(ReduceMeanOpTest, CPUSimple3Axis) {
  Simple3Axis<DeviceType::CPU>();
}

TEST_F(ReduceMeanOpTest, CPUSimpleReduceDims) {
  Simple3D<CPU>({2, 3, 4},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                {0, 1},
                {4},
                {10, 11, 12, 13},
                false);
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const std::vector<index_t> &input_shape,
                const std::vector<int> &axis) {
  testing::internal::LogToStderr();
  srand(time(NULL));
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);
  OpDefBuilder("ReduceMean", "ReduceMeanTest")
      .Input("InputNCHW")
      .AddIntsArg("axis", axis)
      .AddIntArg("keepdims", 1)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                  "Output", NHWC);
  OpDefBuilder("ReduceMean", "ReduceMeanTest")
      .Input("Input")
      .AddIntsArg("axis", axis)
      .AddIntArg("keepdims", 1)
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
}
}  // namespace

TEST_F(ReduceMeanOpTest, GPURandomFloat) {
  RandomTest<DeviceType::GPU, float>({4, 64, 64, 3}, {1, 2});
  RandomTest<DeviceType::GPU, float>({2, 64, 64, 4}, {1, 2});
  RandomTest<DeviceType::GPU, float>({8, 128, 128, 64}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 640, 480, 64}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 480, 640, 32}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 512, 512, 16}, {1, 2});
  RandomTest<DeviceType::GPU, float>({8, 117, 87, 33}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 619, 450, 61}, {1, 2});
  RandomTest<DeviceType::GPU, float>({1, 511, 561, 11}, {1, 2});
}

TEST_F(ReduceMeanOpTest, GPURandomHalf) {
  RandomTest<DeviceType::GPU, half>({4, 64, 64, 3}, {1, 2});
  RandomTest<DeviceType::GPU, half>({2, 64, 64, 4}, {1, 2});
  RandomTest<DeviceType::GPU, half>({8, 128, 128, 64}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 640, 480, 64}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 480, 640, 32}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 512, 512, 16}, {1, 2});
  RandomTest<DeviceType::GPU, half>({8, 117, 87, 33}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 619, 450, 61}, {1, 2});
  RandomTest<DeviceType::GPU, half>({1, 511, 561, 11}, {1, 2});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
