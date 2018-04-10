//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class FullyConnectedOpTest : public OpsTestBase {};

template<DeviceType D>
void Simple(const std::vector<index_t> &input_shape,
            const std::vector<float> &input_value,
            const std::vector<index_t> &weight_shape,
            const std::vector<float> &weight_value,
            const std::vector<index_t> &bias_shape,
            const std::vector<float> &bias_value,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output_value) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input_value);
  net.AddInputFromArray<D, float>("Weight", weight_shape, weight_value);
  net.AddInputFromArray<D, float>("Bias", bias_shape, bias_value);

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Weight", "WeightImage",
                            kernels::BufferType::WEIGHT_HEIGHT);
    BufferToImage<D, float>(&net, "Bias", "BiasImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("FC", "FullyConnectedTest")
      .Input("InputImage")
      .Input("WeightImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .AddIntArg("weight_type", kernels::BufferType::WEIGHT_HEIGHT)
      .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("FC", "FullyConnectedTest")
      .Input("Input")
      .Input("Weight")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<float>(output_shape, output_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(FullyConnectedOpTest, SimpleCPU) {
  Simple<DeviceType::CPU>({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 8},
                          {1, 2, 3, 4, 5, 6, 7, 8}, {1}, {2}, {1, 1, 1, 1},
                          {206});
  Simple<DeviceType::CPU>(
    {1, 1, 2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
    {2}, {2, 3}, {1, 1, 1, 2}, {387, 3853});
  Simple<DeviceType::CPU>(
    {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {5, 6},
    {1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 1, 2, 3,
     4, 5, 6, 10, 20, 30, 40, 50, 60, 1, 2, 3, 4, 5, 6},
    {5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5}, {92, 912, 94, 914, 96});
}

TEST_F(FullyConnectedOpTest, SimpleCPUWithBatch) {
  Simple<DeviceType::CPU>({2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 4},
                          {1, 2, 3, 4}, {1}, {2}, {2, 1, 1, 1}, {32, 72});
}

TEST_F(FullyConnectedOpTest, SimpleOPENCL) {
  Simple<DeviceType::OPENCL>({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 8},
                             {1, 2, 3, 4, 5, 6, 7, 8}, {1}, {2}, {1, 1, 1, 1},
                             {206});
  Simple<DeviceType::OPENCL>(
    {1, 1, 2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
    {2}, {2, 3}, {1, 1, 1, 2}, {387, 3853});
  Simple<DeviceType::OPENCL>(
    {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {5, 6},
    {1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 1, 2, 3,
     4, 5, 6, 10, 20, 30, 40, 50, 60, 1, 2, 3, 4, 5, 6},
    {5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5}, {92, 912, 94, 914, 96});
}

TEST_F(FullyConnectedOpTest, SimpleGPUWithBatch) {
  Simple<DeviceType::OPENCL>({2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 4},
                             {1, 2, 3, 4}, {1}, {2}, {2, 1, 1, 1}, {32, 72});
}

template<typename T>
void Complex(const index_t batch,
             const index_t height,
             const index_t width,
             const index_t channels,
             const index_t out_channel) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("Input")
    .Input("Weight")
    .Input("Bias")
    .Output("Output")
    .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
    "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>(
    "Weight", {out_channel, height * width * channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Bias", {out_channel});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, T>(&net, "Input", "InputImage",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, T>(&net, "Weight", "WeightImage",
                                       kernels::BufferType::WEIGHT_HEIGHT);
  BufferToImage<DeviceType::OPENCL, float>(&net, "Bias", "BiasImage",
                                           kernels::BufferType::ARGUMENT);

  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("InputImage")
    .Input("WeightImage")
    .Input("BiasImage")
    .Output("OutputImage")
    .AddIntArg("weight_type", kernels::BufferType::WEIGHT_HEIGHT)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
    .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(&net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-3);
  }
}

TEST_F(FullyConnectedOpTest, OPENCLAlignedWithoutBatch) {
  Complex<float>(1, 16, 16, 32, 16);
  Complex<float>(1, 16, 32, 32, 32);
}
TEST_F(FullyConnectedOpTest, OPENCLUnAlignedWithoutBatch) {
  Complex<float>(1, 13, 11, 11, 17);
  Complex<float>(1, 23, 29, 23, 113);
}
TEST_F(FullyConnectedOpTest, OPENCLUnAlignedWithBatch) {
  Complex<float>(16, 11, 13, 23, 17);
  Complex<float>(31, 13, 11, 29, 113);
}
TEST_F(FullyConnectedOpTest, OPENCLHalfAlignedWithoutBatch) {
  Complex<half>(1, 16, 16, 32, 16);
  Complex<half>(1, 16, 32, 32, 32);
}
TEST_F(FullyConnectedOpTest, OPENCLHalfUnAlignedWithBatch) {
  Complex<half>(2, 11, 13, 61, 17);
  Complex<half>(16, 13, 12, 31, 113);
  Complex<half>(31, 21, 11, 23, 103);
}

template<typename T>
void TestWXFormat(const index_t batch,
                  const index_t height,
                  const index_t width,
                  const index_t channels,
                  const index_t out_channel) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("Input")
    .Input("Weight")
    .Input("Bias")
    .Output("Output")
    .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
    "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>(
    "Weight", {out_channel, height * width * channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Bias", {out_channel});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, T>(&net, "Input", "InputImage",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, T>(&net, "Weight", "WeightImage",
                                       kernels::BufferType::WEIGHT_WIDTH);
  BufferToImage<DeviceType::OPENCL, T>(&net, "Bias", "BiasImage",
                                           kernels::BufferType::ARGUMENT);

  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("InputImage")
    .Input("WeightImage")
    .Input("BiasImage")
    .Output("OutputImage")
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
    .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(&net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2);
  }
}

TEST_F(FullyConnectedOpTest, OPENCLWidthFormatAligned) {
  TestWXFormat<float>(1, 7, 7, 32, 16);
  TestWXFormat<float>(1, 7, 7, 512, 128);
  TestWXFormat<float>(1, 1, 1, 2048, 1024);
}

TEST_F(FullyConnectedOpTest, OPENCLWidthFormatMultiBatch) {
  TestWXFormat<float>(11, 7, 7, 32, 16);
  TestWXFormat<float>(5, 7, 7, 512, 128);
  TestWXFormat<float>(3, 1, 1, 2048, 1024);
}

TEST_F(FullyConnectedOpTest, OPENCLHalfWidthFormatAligned) {
  TestWXFormat<float>(1, 2, 2, 512, 2);
  TestWXFormat<half>(1, 11, 11, 32, 16);
  TestWXFormat<half>(1, 16, 32, 32, 32);
}

void FullyConnectedTestNEON(const index_t batch,
              const index_t height,
              const index_t width,
              const index_t channels,
              const index_t out_channel) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("Input")
    .Input("Weight")
    .Input("Bias")
    .Output("Output")
    .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>(
    "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::CPU, float>(
    "Weight", {out_channel, height * width * channels});
  net.AddRandomInput<DeviceType::CPU, float>("Bias", {out_channel});

  // run cpu
  net.RunOp();

  // Run on neon
  OpDefBuilder("FC", "FullyConnectedTest")
    .Input("Input")
    .Input("Weight")
    .Input("Bias")
    .Output("OutputNeon")
    .Finalize(net.NewOperatorDef());

  // Run on device
  net.RunOp(DeviceType::NEON);

  net.FillNHWCInputToNCHWInput<DeviceType::CPU, float>("OutputExptected",
                                                       "Output");

  ExpectTensorNear<float>(*net.GetOutput("OutputExptected"),
                          *net.GetOutput("OutputNeon"),
                          0.01);
}

TEST_F(FullyConnectedOpTest, TestNEON) {
  FullyConnectedTestNEON(1, 7, 7, 32, 16);
  FullyConnectedTestNEON(1, 7, 7, 512, 128);
  FullyConnectedTestNEON(1, 1, 1, 2048, 1024);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
