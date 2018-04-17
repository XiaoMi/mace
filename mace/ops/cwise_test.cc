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

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "../kernels/cwise.h"

namespace mace {
namespace ops {
namespace test {

class CWiseOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const kernels::CWiseType type,
            const std::vector<index_t> &shape,
            const std::vector<float> &input0,
            const float x,
            const std::vector<float> &output) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input1", shape, input0);

  if (D == DeviceType::CPU) {
    OpDefBuilder("CWise", "CWiseTest")
        .Input("Input1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("x", x)
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    BufferToImage<D, half>(&net, "Input1", "InputImg1",
                           kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("CWise", "CWiseTest")
        .Input("InputImg1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("x", x)
        .Output("OutputImg")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImg", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  }

  auto expected = CreateTensor<float>(shape, output);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5, 1e-3);
}
}  // namespace

TEST_F(CWiseOpTest, CPUSimple) {
  Simple<DeviceType::CPU>(kernels::CWiseType::MUL, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {0.1, 0.2, .3, .4, .5, .6});

  Simple<DeviceType::CPU>(kernels::CWiseType::ADD, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {3, 4, 5, 6, 7, 8});

  Simple<DeviceType::CPU>(kernels::CWiseType::DIV, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {10, 20, 30, 40, 50, 60});

  Simple<DeviceType::CPU>(kernels::CWiseType::SUB, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, 0, 1, 2, 3, 4});

  Simple<DeviceType::CPU>(kernels::CWiseType::NEG, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, -2, -3, -4, -5, -6});

  Simple<DeviceType::CPU>(kernels::CWiseType::ABS, {1, 1, 2, 3},
                    {1, -2, -0.0001, 4, 5, 6}, 2.0, {1, 2, 0.0001, 4, 5, 6});
}

TEST_F(CWiseOpTest, GPUSimple) {
  Simple<DeviceType::OPENCL>(kernels::CWiseType::MUL, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {0.1, 0.2, .3, .4, .5, .6});

  Simple<DeviceType::OPENCL>(kernels::CWiseType::ADD, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {3, 4, 5, 6, 7, 8});

  Simple<DeviceType::OPENCL>(kernels::CWiseType::DIV, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {10, 20, 30, 40, 50, 60});

  Simple<DeviceType::OPENCL>(kernels::CWiseType::SUB, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, 0, 1, 2, 3, 4});

  Simple<DeviceType::OPENCL>(kernels::CWiseType::NEG, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, -2, -3, -4, -5, -6});

  Simple<DeviceType::OPENCL>(kernels::CWiseType::ABS, {1, 1, 2, 3},
                    {1, -2, -0.0001, 4, 5, 6}, 2.0, {1, 2, 0.0001, 4, 5, 6});
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const kernels::CWiseType type,
                const std::vector<index_t> &shape) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input1", shape);

  OpDefBuilder("CWise", "CWiseTest")
      .Input("Input1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("x", 1.2)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  BufferToImage<D, T>(&net, "Input1", "InputImg1",
                      kernels::BufferType::IN_OUT_CHANNEL);

  OpDefBuilder("CWise", "CWiseTest")
      .Input("InputImg1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("x", 1.2)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("OutputImg")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  ImageToBuffer<D, float>(&net, "OutputImg", "OPENCLOutput",
                          kernels::BufferType::IN_OUT_CHANNEL);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-5, 1e-4);
  } else {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-2, 1e-2);
  }
}
}  // namespace

TEST_F(CWiseOpTest, OPENCLRandomFloat) {
  RandomTest<DeviceType::OPENCL, float>(kernels::CWiseType::MUL,
                                        {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::CWiseType::ADD,
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::CWiseType::SUB,
                                        {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::CWiseType::DIV,
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::CWiseType::NEG,
                                        {13, 32, 32, 64});
}

TEST_F(CWiseOpTest, OPENCLRandomHalf) {
  RandomTest<DeviceType::OPENCL, half>(kernels::CWiseType::MUL,
                                       {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::CWiseType::ADD,
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::CWiseType::SUB,
                                       {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::CWiseType::DIV,
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::CWiseType::NEG,
                                       {13, 32, 32, 64});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
