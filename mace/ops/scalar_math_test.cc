//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "../kernels/scalar_math.h"

namespace mace {
namespace ops {
namespace test {

class ScalarMathOpTest : public OpsTestBase {};


template <DeviceType D>
void Simple(const kernels::ScalarMathType type,
            const std::vector<index_t> &shape,
            const std::vector<float> &input0,
            const float x,
            const std::vector<float> &output) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input1", shape, input0);

  if (D == DeviceType::CPU) {
    OpDefBuilder("ScalarMath", "ScalarMathTest")
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
    OpDefBuilder("ScalarMath", "ScalarMathTest")
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

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-3);
}

TEST_F(ScalarMathOpTest, CPUSimple) {
  Simple<DeviceType::CPU>(kernels::ScalarMathType::MUL, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {0.1, 0.2, .3, .4, .5, .6});

  Simple<DeviceType::CPU>(kernels::ScalarMathType::ADD, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {3, 4, 5, 6, 7, 8});

  Simple<DeviceType::CPU>(kernels::ScalarMathType::DIV, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {10, 20, 30, 40, 50, 60});

  Simple<DeviceType::CPU>(kernels::ScalarMathType::SUB, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, 0, 1, 2, 3, 4});
}

TEST_F(ScalarMathOpTest, GPUSimple) {
  Simple<DeviceType::OPENCL>(kernels::ScalarMathType::MUL, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {0.1, 0.2, .3, .4, .5, .6});

  Simple<DeviceType::OPENCL>(kernels::ScalarMathType::ADD, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {3, 4, 5, 6, 7, 8});

  Simple<DeviceType::OPENCL>(kernels::ScalarMathType::DIV, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 0.1, {10, 20, 30, 40, 50, 60});

  Simple<DeviceType::OPENCL>(kernels::ScalarMathType::SUB, {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6}, 2.0, {-1, 0, 1, 2, 3, 4});
}

template <DeviceType D, typename T>
void RandomTest(const kernels::ScalarMathType type,
                const std::vector<index_t> &shape) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input1", shape);

  OpDefBuilder("ScalarMath", "ScalarMathTest")
      .Input("Input1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("x", 1.2)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  BufferToImage<D, T>(&net, "Input1", "InputImg1",
                      kernels::BufferType::IN_OUT_CHANNEL);

  OpDefBuilder("ScalarMath", "ScalarMathTest")
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
                            *net.GetOutput("OPENCLOutput"), 1e-3);
  } else {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-1);
  }
}

TEST_F(ScalarMathOpTest, OPENCLRandomFloat) {
  RandomTest<DeviceType::OPENCL, float>(kernels::ScalarMathType::MUL,
                                        {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::ScalarMathType::ADD,
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::ScalarMathType::SUB,
                                        {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::ScalarMathType::DIV,
                                        {13, 32, 32, 64});
}

TEST_F(ScalarMathOpTest, OPENCLRandomHalf) {
  RandomTest<DeviceType::OPENCL, half>(kernels::ScalarMathType::MUL,
                                       {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::ScalarMathType::ADD,
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::ScalarMathType::SUB,
                                       {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::ScalarMathType::DIV,
                                       {13, 32, 32, 64});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
