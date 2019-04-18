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

#ifdef MACE_ENABLE_OPENCL

#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/opencl/buffer_transformer.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void TestBidirectionTransform(const OpenCLBufferType type,
                              const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  net.AddRandomInput<D, T>("Input", input_shape);
  Tensor *b2i_output = net.ws()->CreateTensor(
      "B2IOutput", context.device()->allocator(), DataTypeToEnum<T>::value);

  OpenCLBufferTransformer<T>(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
      .Transform(&context, net.ws()->GetTensor("Input"),
                 type, MemoryType::GPU_IMAGE, 0, b2i_output);

  // Inverse Transform
  Tensor *i2b_output = net.ws()->CreateTensor(
      "I2BOutput", context.device()->allocator(), DataTypeToEnum<T>::value);
  OpenCLBufferTransformer<T>(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER)
      .Transform(&context, b2i_output,
                 type, MemoryType::GPU_BUFFER, 0, i2b_output);

  // Check
  ExpectTensorNear<T>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"),
                      1e-5);
}
}  // namespace

TEST(BufferToImageTest, ArgSmall) {
  TestBidirectionTransform<DeviceType::GPU, float>(OpenCLBufferType::ARGUMENT,
                                                   {1});
}

TEST(BufferToImageTest, ArgHalfSmall) {
  TestBidirectionTransform<DeviceType::GPU, half>(OpenCLBufferType::ARGUMENT,
                                                  {11});
}

TEST(BufferToImageTest, ArgMedium) {
  TestBidirectionTransform<DeviceType::GPU, float>(OpenCLBufferType::ARGUMENT,
                                                   {11});
}

TEST(BufferToImageTest, ArgLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(OpenCLBufferType::ARGUMENT,
                                                   {256});
}

TEST(BufferToImageTest, InputSmallSingleChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::IN_OUT_CHANNEL, {1, 2, 3, 1});
}

TEST(BufferToImageTest, InputSmallMultipleChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::IN_OUT_CHANNEL, {1, 2, 3, 3});
}

TEST(BufferToImageTest, InputSmallMultipleBatchAndChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::IN_OUT_CHANNEL, {3, 2, 3, 3});
}

TEST(BufferToImageTest, InputMedium) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::IN_OUT_CHANNEL, {3, 13, 17, 128});
}

TEST(BufferToImageTest, InputLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::IN_OUT_CHANNEL, {3, 64, 64, 256});
}

TEST(BufferToImageTest, Filter1x1Small) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {5, 3, 1, 1});
}

TEST(BufferToImageTest, Filter1x1Medium) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {13, 17, 1, 1});
}

TEST(BufferToImageTest, Filter1x1Large) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {512, 128, 1, 1});
}

TEST(BufferToImageTest, Filter3x3Small) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {3, 5, 3, 3});
}

TEST(BufferToImageTest, Filter3x3Medium) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {17, 13, 3, 3});
}

TEST(BufferToImageTest, Filter3x3Large) {
  TestBidirectionTransform<DeviceType::GPU, float>(CONV2D_FILTER,
                                                   {256, 128, 3, 3});
}

TEST(BufferToImageTest, WeightWidthSmall) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_WIDTH,
      {1, 3, 3, 3});
}

TEST(BufferToImageTest, WeightWidthMedium) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_WIDTH,
      {11, 13, 13, 17});
}

TEST(BufferToImageTest, WeightWidthLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_WIDTH,
      {64, 64, 11, 13});
}

TEST(BufferToImageTest, WeightHeightSmall) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_HEIGHT,
      {2, 1, 1, 1});
}

TEST(BufferToImageTest, WeightHeightMedium) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_HEIGHT,
      {11, 13, 13, 17});
}

TEST(BufferToImageTest, WeightHeightLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(
      OpenCLBufferType::WEIGHT_HEIGHT,
      {64, 16, 11, 13});
}

namespace {
template <DeviceType D, typename T>
void TestDiffTypeBidirectionTransform(const OpenCLBufferType type,
                                      const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);
  Tensor *b2i_output = net.ws()->CreateTensor(
      "B2IOutput", context.device()->allocator(), DataTypeToEnum<T>::value);

  OpenCLBufferTransformer<T>(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
      .Transform(&context, net.ws()->GetTensor("Input"),
                 type, MemoryType::GPU_IMAGE, 0, b2i_output);

  // Inverse Transform
  Tensor *i2b_output = net.ws()->CreateTensor(
      "I2BOutput", context.device()->allocator(), DT_FLOAT);
  OpenCLBufferTransformer<float>(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER)
      .Transform(&context, b2i_output,
                 type, MemoryType::GPU_BUFFER, 0, i2b_output);

  // Check
  ExpectTensorNear<float>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"),
                          1e-3, 1e-6);
}
}  // namespace

TEST(BufferToImageTest, ArgFloatToHalfSmall) {
  TestDiffTypeBidirectionTransform<DeviceType::GPU, half>(
      OpenCLBufferType::ARGUMENT,
      {11});
}

namespace {
template <DeviceType D, typename T>
void TestStringHalfBidirectionTransform(const OpenCLBufferType type,
                                        const std::vector<index_t> &input_shape,
                                        const unsigned char *input_data) {
  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  const half *h_data = reinterpret_cast<const half *>(input_data);
  net.AddInputFromArray<D, half>("Input", input_shape,
                                 std::vector<half>(h_data, h_data + 2));
  Tensor *b2i_output = net.ws()->CreateTensor(
      "B2IOutput", context.device()->allocator(), DataTypeToEnum<T>::value);

  // Transform
  OpenCLBufferTransformer<T>(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
      .Transform(&context, net.ws()->GetTensor("Input"),
                 type, MemoryType::GPU_IMAGE, 0, b2i_output);

  // Inverse Transform
  Tensor *i2b_output = net.ws()->CreateTensor(
      "I2BOutput", context.device()->allocator(), DataTypeToEnum<T>::value);
  OpenCLBufferTransformer<T>(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER)
      .Transform(&context, b2i_output,
                 type, MemoryType::GPU_BUFFER, 0, i2b_output);

  // Check
  ExpectTensorNear<half>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"),
                         1e-3, 1e-6);
}
}  // namespace

TEST(BufferToImageTest, ArgStringHalfToHalfSmall) {
  const unsigned char input_data[] = {
      0xCD, 0x3C, 0x33, 0x40,
  };
  TestStringHalfBidirectionTransform<DeviceType::GPU, half>(
      OpenCLBufferType::ARGUMENT, {2}, input_data);
}

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_ENABLE_OPENCL
