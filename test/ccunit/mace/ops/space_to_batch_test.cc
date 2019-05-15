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

#include <fstream>

#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D>
void RunSpaceToBatch(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const std::vector<int> &block_shape_data,
                     const std::vector<int> &padding_data,
                     const Tensor *expected) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  if (D == GPU) {
    OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("paddings", padding_data)
        .AddIntsArg("block_shape", block_shape_data)
        .Finalize(net.NewOperatorDef());
  } else if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntsArg("paddings", padding_data)
        .AddIntsArg("block_shape", block_shape_data)
        .Finalize(net.NewOperatorDef());
  }

  // Run
  net.RunOp(D);

  if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }
  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}

template <DeviceType D>
void RunBatchToSpace(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const std::vector<int> &block_shape_data,
                     const std::vector<int> &crops_data,
                     const Tensor *expected) {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  if (D == GPU) {
    OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("crops", crops_data)
        .AddIntsArg("block_shape", block_shape_data)
        .Finalize(net.NewOperatorDef());
  } else if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntsArg("crops", crops_data)
        .AddIntsArg("block_shape", block_shape_data)
        .Finalize(net.NewOperatorDef());
  }

  // Run
  net.RunOp(D);

  if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }
  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}

template <typename T>
void TestBidirectionalTransform(const std::vector<index_t> &space_shape,
                                const std::vector<float> &space_data,
                                const std::vector<int> &block_data,
                                const std::vector<int> &padding_data,
                                const std::vector<index_t> &batch_shape,
                                const std::vector<float> &batch_data) {
  OpsTestNet net;
  auto space_tensor = net.CreateTensor<T, GPU>();
  space_tensor->Resize(space_shape);
  {
    Tensor::MappingGuard space_mapper(space_tensor.get());
    T *space_ptr = space_tensor->template mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(space_tensor->size()) == space_data.size())
        << "Space tensor size:" << space_tensor->size()
        << ", space data size:" << space_data.size();
    memcpy(space_ptr, space_data.data(), space_data.size() * sizeof(T));
  }

  auto batch_tensor = net.CreateTensor<T, GPU>();
  batch_tensor->Resize(batch_shape);
  {
    Tensor::MappingGuard batch_mapper(batch_tensor.get());
    T *batch_ptr = batch_tensor->template mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(batch_tensor->size()) == batch_data.size());
    memcpy(batch_ptr, batch_data.data(), batch_data.size() * sizeof(T));
  }

  RunSpaceToBatch<DeviceType::GPU>(space_shape, space_data, block_data,
                                   padding_data, batch_tensor.get());
  RunSpaceToBatch<DeviceType::CPU>(space_shape, space_data, block_data,
                                   padding_data, batch_tensor.get());

  RunBatchToSpace<DeviceType::GPU>(batch_shape, batch_data, block_data,
                                   padding_data, space_tensor.get());
  RunBatchToSpace<DeviceType::CPU>(batch_shape, batch_data, block_data,
                                   padding_data, space_tensor.get());
}


void TestSpaceToBatchLargeInput(const std::vector<index_t> &input_shape,
                                const std::vector<int> &block_shape_data,
                                const std::vector<int> &padding_data) {
  OpsTestNet net;
  net.AddRandomInput<GPU, float>("Input", input_shape);

  // run gpu
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("Input")
      .Output("OutputGPU")
      .AddIntsArg("paddings", padding_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(GPU);

  // run cpu
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("paddings", padding_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "OutputCPU", DataFormat::NHWC);

  // Check
  ExpectTensorNear<float>(*net.GetOutput("OutputCPU"),
                          *net.GetOutput("OutputGPU"));
}

void TestoBatchToSpaceLargeInput(const std::vector<index_t> &input_shape,
                                 const std::vector<int> &block_shape_data,
                                 const std::vector<int> &crops_data) {
  OpsTestNet net;
  net.AddRandomInput<GPU, float>("Input", input_shape);

  // run gpu
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("Input")
      .Output("OutputGPU")
      .AddIntsArg("crops", crops_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(GPU);

  // run cpu
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("crops", crops_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "OutputCPU", DataFormat::NHWC);

  // Check
  ExpectTensorNear<float>(*net.GetOutput("OutputCPU"),
                          *net.GetOutput("OutputGPU"));
}

void TestSpaceToBatchQuantize(const std::vector<index_t> &input_shape,
                              const std::vector<int> &block_shape_data,
                              const std::vector<int> &padding_data) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input",
                                 input_shape,
                                 false,
                                 false,
                                 true,
                                 -1.f,
                                 1.f);

  // run cpu
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("paddings", padding_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "OutputCPU", DataFormat::NHWC);

  // run quantize
  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("QuantizedInput")
      .Output("QuantizedOutput")
      .AddIntsArg("paddings", padding_data)
      .AddIntsArg("block_shape", block_shape_data)
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  Tensor *eq_output = net.GetTensor("QuantizedInput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("OutputCPU"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}

void TestoBatchToSpaceQuantize(const std::vector<index_t> &input_shape,
                               const std::vector<int> &block_shape_data,
                               const std::vector<int> &crops_data) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input",
                                 input_shape,
                                 false,
                                 false,
                                 true,
                                 -1.f,
                                 1.f);

  // run cpu
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("crops", crops_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "OutputCPU", DataFormat::NHWC);

  // run quantize
  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("QuantizedInput")
      .Output("QuantizedOutput")
      .AddIntsArg("crops", crops_data)
      .AddIntsArg("block_shape", block_shape_data)
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  Tensor *eq_output = net.GetTensor("QuantizedInput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("OutputCPU"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}

}  // namespace

TEST(SpaceToBatchTest, SmallData) {
  TestBidirectionalTransform<float>({1, 2, 2, 1}, {1, 2, 3, 4}, {2, 2},
                                    {0, 0, 0, 0}, {4, 1, 1, 1}, {1, 2, 3, 4});
}

TEST(SpaceToBatchTest, SmallDataWithOnePadding) {
  TestBidirectionalTransform<float>({1, 2, 2, 1}, {1, 2, 3, 4}, {3, 3},
                                    {1, 0, 1, 0}, {9, 1, 1, 1},
                                    {0, 0, 0, 0, 1, 2, 0, 3, 4});
}

TEST(SpaceToBatchTest, SmallDataWithTwoPadding) {
  TestBidirectionalTransform<float>(
      {1, 2, 2, 1}, {1, 2, 3, 4}, {2, 2}, {1, 1, 1, 1}, {4, 2, 2, 1},
      {0, 0, 0, 4, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0});
}

TEST(SpaceToBatchTest, SmallDataWithLargeImage) {
  TestBidirectionalTransform<float>(
      {1, 2, 10, 1},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
      {2, 2}, {0, 0, 0, 0}, {4, 1, 5, 1},
      {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 12, 14, 16, 18, 20});
}

TEST(SpaceToBatchTest, MultiChannelData) {
  TestBidirectionalTransform<float>(
      {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2},
      {0, 0, 0, 0}, {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

TEST(SpaceToBatchTest, LargerMultiChannelData) {
  TestBidirectionalTransform<float>(
      {1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {2, 2}, {0, 0, 0, 0}, {4, 2, 2, 1},
      {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16});
}

TEST(SpaceToBatchTest, MultiBatchData) {
  TestBidirectionalTransform<float>(
      {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {2, 2}, {0, 0, 0, 0}, {8, 1, 2, 1},
      {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16});
}

TEST(SpaceToBatchTest, MultiBatchAndChannelData) {
  TestBidirectionalTransform<float>(
      {2, 2, 4, 2},
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
      {2, 2}, {0, 0, 0, 0}, {8, 1, 2, 2},
      {1, 2,  5,  6,  17, 18, 21, 22, 3,  4,  7,  8,  19, 20, 23, 24,
       9, 10, 13, 14, 25, 26, 29, 30, 11, 12, 15, 16, 27, 28, 31, 32});
}

TEST(SpaceToBatchTest, LargeData) {
  TestSpaceToBatchLargeInput({1, 256, 256, 32}, {8, 8}, {0, 0, 0, 0});
  TestSpaceToBatchLargeInput({1, 256, 256, 32}, {8, 8}, {4, 4, 4, 4});
  TestoBatchToSpaceLargeInput({64, 32, 32, 32}, {8, 8}, {0, 0, 0, 0});
  TestoBatchToSpaceLargeInput({64, 32, 32, 32}, {8, 8}, {4, 4, 4, 4});
}

TEST(SpaceToBatchTest, Quantize) {
  TestSpaceToBatchQuantize({1, 256, 256, 32}, {8, 8}, {0, 0, 0, 0});
  TestSpaceToBatchQuantize({1, 256, 256, 32}, {8, 8}, {4, 4, 4, 4});
  TestoBatchToSpaceQuantize({64, 32, 32, 32}, {8, 8}, {0, 0, 0, 0});
  TestoBatchToSpaceQuantize({64, 32, 32, 32}, {8, 8}, {4, 4, 4, 4});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
