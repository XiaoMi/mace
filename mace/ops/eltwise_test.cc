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

#include "mace/kernels/eltwise.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class EltwiseOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const kernels::EltwiseType type,
            const std::vector<index_t> &shape0,
            const std::vector<index_t> &shape1,
            const std::vector<float> &input0,
            const std::vector<float> &input1,
            const std::vector<float> &output,
            const float x = 1.f,
            const std::vector<float> coeff = {}) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input1", shape0, input0);
  net.AddInputFromArray<D, float>("Input2", shape1, input1);

  if (D == DeviceType::CPU) {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input1")
        .Input("Input2")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("x", x)
        .AddFloatsArg("coeff", coeff)
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    BufferToImage<D, half>(&net, "Input1", "InputImg1",
                           kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(&net, "Input2", "InputImg2",
                           kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("InputImg1")
        .Input("InputImg2")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("x", x)
        .AddFloatsArg("coeff", coeff)
        .Output("OutputImg")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImg", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  }

  auto expected = CreateTensor<float>(shape0, output);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(EltwiseOpTest, CPUSimple) {
  Simple<DeviceType::CPU>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {1, 4, 9, 16, 25, 36});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {2, 4, 6, 8, 10, 12});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {3, 6, 9, 12, 15, 18}, 1., {2, 1});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {1, 2, 3, 4, 6, 6});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {1, 1, 3, 3, 5, 6});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {0, 1, 0, 1, 1, 0});
  Simple<DeviceType::CPU>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 2, 10, 24},
                          {1, 2, 1, 2, 0.5, 0.25});

  Simple<DeviceType::CPU>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {1, 4, 9, 4, 10, 18});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {2, 4, 6, 5, 7, 9});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {3, 6, 9, 9, 12, 15}, 1., {2, 1});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 2, 3, 4, 5, 6});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 1, 3, 1, 1, 3});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {0, 1, 0, 9, 16, 9});
  Simple<DeviceType::CPU>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 2, 1, 4, 5, 2});

  Simple<DeviceType::CPU>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {2, 4, 6, 8, 10, 12}, 2);
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {3, 4, 5, 6, 7, 8}, 2);
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {4, 6, 8, 10, 12, 14}, 2, {2, 1});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {3},
                          {3, 3, 3, 4, 5, 6}, 3);
  Simple<DeviceType::CPU>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {3},
                          {1, 2, 3, 3, 3, 3}, 3);
  Simple<DeviceType::CPU>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {0.5},
                          {2, 4, 6, 8, 10, 12}, 0.5);
  Simple<DeviceType::CPU>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {3},
                          {4, 1, 0, 1, 4, 9}, 3);
}

TEST_F(EltwiseOpTest, GPUSimple) {
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {1, 4, 9, 16, 25, 36});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {2, 4, 6, 8, 10, 12});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6},
                          {3, 6, 9, 12, 15, 18}, 1., {2, 1});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {1, 2, 3, 4, 6, 6});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {1, 1, 3, 3, 5, 6});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 2, 10, 24},
                          {1, 2, 1, 2, 0.5, 0.25});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3, 3, 6, 6},
                          {0, 1, 0, 1, 1, 0});

  Simple<DeviceType::OPENCL>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {1, 4, 9, 4, 10, 18});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {2, 4, 6, 5, 7, 9});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                          {3, 6, 9, 9, 12, 15}, 1., {2, 1});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 2, 3, 4, 5, 6});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 1, 3, 1, 1, 3});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {0, 1, 0, 9, 16, 9});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 1, 3},
                          {1, 2, 3, 4, 5, 6}, {1, 1, 3},
                          {1, 2, 1, 4, 5, 2});

  Simple<DeviceType::OPENCL>(kernels::EltwiseType::PROD, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {2, 4, 6, 8, 10, 12}, 2);
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {3, 4, 5, 6, 7, 8}, 2);
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {2},
                          {4, 6, 8, 10, 12, 14}, 2, {2, 1});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MAX, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {3},
                          {3, 3, 3, 4, 5, 6}, 3);
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MIN, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {3},
                          {1, 2, 3, 3, 3, 3}, 3);
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SQR_DIFF, {1, 1, 2, 3},
                             {1, 1, 1, 1},
                             {1, 2, 3, 4, 5, 6}, {3},
                             {4, 1, 0, 1, 4, 9}, 3);
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::DIV, {1, 1, 2, 3},
                          {1, 1, 1, 1},
                          {1, 2, 3, 4, 5, 6}, {0.5},
                          {2, 4, 6, 8, 10, 12}, 0.5);
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const kernels::EltwiseType type,
                const std::vector<index_t> &shape1,
                const std::vector<index_t> &shape2) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  bool is_divide = (type == kernels::EltwiseType::DIV);

  // Add input data
  net.AddRandomInput<D, float>("Input1", shape1, true, is_divide);
  net.AddRandomInput<D, float>("Input2", shape2, true, is_divide);



  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("Input1")
      .Input("Input2")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", {1.2, 2.1})
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  BufferToImage<D, T>(&net, "Input1", "InputImg1",
                      kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(&net, "Input2", "InputImg2",
                      kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("InputImg1")
      .Input("InputImg2")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", {1.2, 2.1})
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

TEST_F(EltwiseOpTest, OPENCLRandomFloat) {
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::PROD,
                                        {3, 23, 37, 19},
                                        {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SUM,
                                        {13, 32, 32, 64},
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MAX,
                                        {3, 32, 32, 64},
                                        {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MIN,
                                        {13, 32, 32, 64},
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::DIV,
                                        {13, 32, 32, 64},
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SQR_DIFF,
                                        {13, 32, 32, 64},
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::PROD,
                                        {3, 23, 37, 19},
                                        {1, 1, 37, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SUM,
                                        {13, 32, 32, 64},
                                        {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MAX,
                                        {3, 32, 32, 64},
                                        {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MIN,
                                        {13, 32, 32, 64},
                                        {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::DIV,
                                        {13, 32, 32, 63},
                                        {1, 1, 32, 63});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SQR_DIFF,
                                        {13, 32, 32, 64},
                                        {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::PROD,
                                        {3, 23, 37, 19},
                                        {1, 1, 1, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SUM,
                                        {13, 32, 32, 64},
                                        {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MAX,
                                        {3, 32, 32, 64},
                                        {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MIN,
                                        {13, 32, 32, 64},
                                        {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::DIV,
                                        {13, 32, 32, 64},
                                        {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SQR_DIFF,
                                        {13, 32, 32, 64},
                                        {1, 1, 1, 64});
}

TEST_F(EltwiseOpTest, OPENCLRandomHalf) {
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::PROD,
                                       {3, 23, 37, 19},
                                       {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::PROD,
                                       {3, 23, 37, 19},
                                       {1, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::PROD,
                                       {3, 23, 37, 19},
                                       {1, 1, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::PROD,
                                       {3, 23, 37, 19},
                                       {1, 1, 1, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SUM,
                                       {13, 32, 32, 64},
                                       {1, 1, 1, 1});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SUM,
                                       {13, 32, 32, 64},
                                       {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SUM,
                                       {13, 32, 32, 64},
                                       {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::MAX,
                                       {3, 32, 32, 64},
                                       {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::MAX,
                                       {3, 32, 32, 64},
                                       {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::MIN,
                                       {13, 32, 32, 64},
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SQR_DIFF,
                                       {13, 32, 32, 64},
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SQR_DIFF,
                                       {13, 32, 32, 64},
                                       {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SQR_DIFF,
                                       {13, 32, 32, 64},
                                       {1, 1, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::DIV,
                                       {13, 32, 32, 64},
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::DIV,
                                       {13, 32, 32, 64},
                                       {1, 1, 1, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::DIV,
                                       {13, 32, 32, 64},
                                       {1, 1, 32, 64});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
