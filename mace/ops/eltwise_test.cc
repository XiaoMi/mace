//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/kernels/eltwise.h"

namespace mace {

class EltwiseOpTest : public OpsTestBase {};

template<DeviceType D>
void Simple(const kernels::EltwiseType type,
            const std::vector<index_t> &shape,
            const std::vector<float> &input0,
            const std::vector<float> &input1,
            const std::vector<float> &output,
            const std::vector<float> coeff = {}) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input1", shape, input0);
  net.AddInputFromArray<D, float>("Input2", shape, input1);

  if (D == DeviceType::CPU) {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input1")
        .Input("Input2")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", coeff)
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    BufferToImage<D, half>(net, "Input1", "InputImg1", kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(net, "Input2", "InputImg2", kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("InputImg1")
        .Input("InputImg2")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", coeff)
        .Output("OutputImg")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(net, "OutputImg", "Output", kernels::BufferType::IN_OUT_CHANNEL);
  }

  auto expected = CreateTensor<float>(shape, output);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-3);
}

TEST_F(EltwiseOpTest, CPUSimple) {
  Simple<DeviceType::CPU>(kernels::EltwiseType::PROD,
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6},
                          {1, 2, 3, 4, 5, 6},
                          {1, 4, 9, 16, 25, 36});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM,
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6},
                          {1, 2, 3, 4, 5, 6},
                          {2, 4, 6, 8, 10, 12});
  Simple<DeviceType::CPU>(kernels::EltwiseType::SUM,
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6},
                          {1, 2, 3, 4, 5, 6},
                          {3, 6, 9, 12, 15, 18},
                          {2, 1});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MAX,
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6},
                          {1, 1, 3, 3, 6, 6},
                          {1, 2, 3, 4, 6, 6});
  Simple<DeviceType::CPU>(kernels::EltwiseType::MIN,
                          {1, 1, 2, 3},
                          {1, 2, 3, 4, 5, 6},
                          {1, 1, 3, 3, 6, 6},
                          {1, 1, 3, 3, 5, 6});
}

TEST_F(EltwiseOpTest, GPUSimple) {
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::PROD,
                             {1, 1, 2, 3},
                             {1, 2, 3, 4, 5, 6},
                             {1, 2, 3, 4, 5, 6},
                             {1, 4, 9, 16, 25, 36});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM,
                             {1, 1, 2, 3},
                             {1, 2, 3, 4, 5, 6},
                             {1, 2, 3, 4, 5, 6},
                             {2, 4, 6, 8, 10, 12});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::SUM,
                             {1, 1, 2, 3},
                             {1, 2, 3, 4, 5, 6},
                             {1, 2, 3, 4, 5, 6},
                             {3, 6, 9, 12, 15, 18},
                             {2, 1});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MAX,
                             {1, 1, 2, 3},
                             {1, 2, 3, 4, 5, 6},
                             {1, 1, 3, 3, 6, 6},
                             {1, 2, 3, 4, 6, 6});
  Simple<DeviceType::OPENCL>(kernels::EltwiseType::MIN,
                             {1, 1, 2, 3},
                             {1, 2, 3, 4, 5, 6},
                             {1, 1, 3, 3, 6, 6},
                             {1, 1, 3, 3, 5, 6});
}

template<DeviceType D, typename T>
void RandomTest(const kernels::EltwiseType type,
                const std::vector<index_t> &shape) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input1", shape);
  net.AddRandomInput<D, float>("Input2", shape);

  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("Input1")
      .Input("Input2")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", {1.2, 2.1})
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();

  BufferToImage<D, T>(net, "Input1", "InputImg1", kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(net, "Input2", "InputImg2", kernels::BufferType::IN_OUT_CHANNEL);
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

  ImageToBuffer<D, float>(net, "OutputImg", "OPENCLOutput", kernels::BufferType::IN_OUT_CHANNEL);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*net.GetTensor("Output"), *net.GetOutput("OPENCLOutput"), 1e-3);
  } else {
    ExpectTensorNear<float>(*net.GetTensor("Output"), *net.GetOutput("OPENCLOutput"), 1e-1);
  }
}

TEST_F(EltwiseOpTest, OPENCLRandomFloat) {
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::PROD,
                                        {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::SUM,
                                        {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MAX,
                                        {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, float>(kernels::EltwiseType::MIN,
                                        {13, 32, 32, 64});
}

TEST_F(EltwiseOpTest, OPENCLRandomHalf) {
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::PROD,
                                       {3, 23, 37, 19});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::SUM,
                                       {13, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::MAX,
                                       {3, 32, 32, 64});
  RandomTest<DeviceType::OPENCL, half>(kernels::EltwiseType::MIN,
                                       {13, 32, 32, 64});
}

}  // namespace mace
