//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class FoldedBatchNormOpTest : public OpsTestBase {};

void CalculateScaleOffset(const std::vector<float> &gamma,
                          const std::vector<float> &beta,
                          const std::vector<float> &mean,
                          const std::vector<float> &var,
                          const float epsilon,
                          std::vector<float> *scale,
                          std::vector<float> *offset) {
  size_t size = gamma.size();
  for (int i = 0; i < size; ++i) {
    (*scale)[i] = gamma[i] / std::sqrt(var[i] + epsilon);
    (*offset)[i] = (*offset)[i] - mean[i] * (*scale)[i];
  }
}

template <DeviceType D>
void Simple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  std::vector<float> scale(1);
  std::vector<float> offset(1);
  CalculateScaleOffset({4.0f}, {2.0}, {10}, {11.67f}, 1e-3, &scale, &offset);
  net.AddInputFromArray<D, float>("Scale", {1}, scale);
  net.AddInputFromArray<D, float>("Offset", {1}, offset);

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(net, "Scale", "ScaleImage",
                            kernels::BufferType::ARGUMENT);
    BufferToImage<D, float>(net, "Offset", "OffsetImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
        .Input("InputImage")
        .Input("ScaleImage")
        .Input("OffsetImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
        .Input("Input")
        .Input("Scale")
        .Input("Offset")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected =
      CreateTensor<float>({1, 6, 2, 1}, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                         3.17, 3.17, 5.51, 5.51, 7.86, 7.86});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(FoldedBatchNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

/*
TEST_F(FoldedBatchNormOpTest, SimpleNEON) {
  Simple<DeviceType::NEON>();
}
*/

TEST_F(FoldedBatchNormOpTest, SimpleOPENCL) { Simple<DeviceType::OPENCL>(); }

/*
TEST_F(FoldedBatchNormOpTest, SimpleRandomNeon) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 64;
  index_t width = 64;
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", {batch, channels, height,
width});
  net.AddRandomInput<DeviceType::CPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::CPU, float>("Epsilon", {}, {1e-3});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run NEON
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}

TEST_F(FoldedBatchNormOpTest, ComplexRandomNeon) {
  srand(time(NULL));

  // generate random input
  index_t batch = 1 + rand() % 10;
  index_t channels = 3 + rand() % 50;
  index_t height = 103;
  index_t width = 113;
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .Input("Epsilon")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", {batch, channels, height,
width});
  net.AddRandomInput<DeviceType::CPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Offset", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Mean", {channels});
  net.AddRandomInput<DeviceType::CPU, float>("Var", {channels}, true);
  net.AddInputFromArray<DeviceType::CPU, float>("Epsilon", {}, {1e-3});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run NEON
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-2);
}
*/

TEST_F(FoldedBatchNormOpTest, SimpleRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64;
  index_t width = 64;

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, float>(net, "Input", "InputImage",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, float>(net, "Scale", "ScaleImage",
                                           kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::OPENCL, float>(net, "Offset", "OffsetImage",
                                           kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);
  net.Sync();

  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2);
}

TEST_F(FoldedBatchNormOpTest, SimpleRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64;
  index_t width = 64;

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, half>(net, "Input", "InputImage",
                                          kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, half>(net, "Scale", "ScaleImage",
                                          kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::OPENCL, half>(net, "Offset", "OffsetImage",
                                          kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);
  net.Sync();

  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.5);
}

TEST_F(FoldedBatchNormOpTest, ComplexRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, float>(net, "Input", "InputImage",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, float>(net, "Scale", "ScaleImage",
                                           kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::OPENCL, float>(net, "Offset", "OffsetImage",
                                           kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2);
}

TEST_F(FoldedBatchNormOpTest, ComplexRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "Input", {batch, height, width, channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::OPENCL, float>("Offset", {channels});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, half>(net, "Input", "InputImage",
                                          kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::OPENCL, half>(net, "Scale", "ScaleImage",
                                          kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::OPENCL, half>(net, "Offset", "OffsetImage",
                                          kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
