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

#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void TestBidirectionTransform(const int type,
                              const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpDefBuilder("BufferToImage", "BufferToImageTest")
      .Input("Input")
      .Output("B2IOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, T>("Input", input_shape);

  // Run
  net.RunOp(D);

  OpDefBuilder("ImageToBuffer", "ImageToBufferTest")
      .Input("B2IOutput")
      .Output("I2BOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  // Check
  ExpectTensorNear<T>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"),
                      1e-5);
}
}  // namespace

TEST(BufferToImageTest, ArgSmall) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::ARGUMENT, {1});
}

TEST(BufferToImageTest, ArgHalfSmall) {
  TestBidirectionTransform<DeviceType::GPU, half>(kernels::ARGUMENT, {11});
}

TEST(BufferToImageTest, ArgMedia) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::ARGUMENT, {11});
}

TEST(BufferToImageTest, ArgLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::ARGUMENT, {256});
}

TEST(BufferToImageTest, InputSmallSingleChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::IN_OUT_CHANNEL,
                                                      {1, 2, 3, 1});
}

TEST(BufferToImageTest, InputSmallMultipleChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::IN_OUT_CHANNEL,
                                                      {1, 2, 3, 3});
}

TEST(BufferToImageTest, InputSmallMultipleBatchAndChannel) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::IN_OUT_CHANNEL,
                                                      {3, 2, 3, 3});
}

TEST(BufferToImageTest, InputMedia) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::IN_OUT_CHANNEL,
                                                      {3, 13, 17, 128});
}

TEST(BufferToImageTest, InputLarge) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::IN_OUT_CHANNEL,
                                                      {3, 64, 64, 256});
}

TEST(BufferToImageTest, Filter1x1Small) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {1, 1, 3, 5});
}

TEST(BufferToImageTest, Filter1x1Media) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {1, 1, 13, 17});
}

TEST(BufferToImageTest, Filter1x1Large) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {1, 1, 128, 512});
}

TEST(BufferToImageTest, Filter3x3Small) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {3, 3, 3, 5});
}

TEST(BufferToImageTest, Filter3x3Meida) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {3, 3, 13, 17});
}

TEST(BufferToImageTest, Filter3x3Large) {
  TestBidirectionTransform<DeviceType::GPU, float>(kernels::CONV2D_FILTER,
                                                      {3, 3, 128, 256});
}

namespace {
template <DeviceType D, typename T>
void TestDiffTypeBidirectionTransform(const int type,
                                      const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpDefBuilder("BufferToImage", "BufferToImageTest")
      .Input("Input")
      .Output("B2IOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  // Run
  net.RunOp(D);

  OpDefBuilder("ImageToBuffer", "ImageToBufferTest")
      .Input("B2IOutput")
      .Output("I2BOutput")
      .AddIntArg("buffer_type", type)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  // Check
  ExpectTensorNear<float>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"),
                          1e-3, 1e-6);
}
}  // namespace

TEST(BufferToImageTest, ArgFloatToHalfSmall) {
  TestDiffTypeBidirectionTransform<DeviceType::GPU, half>(kernels::ARGUMENT,
                                                             {11});
}

namespace {
template <DeviceType D, typename T>
void TestStringHalfBidirectionTransform(const int type,
                                        const std::vector<index_t> &input_shape,
                                        const unsigned char *input_data) {
  OpsTestNet net;
  OpDefBuilder("BufferToImage", "BufferToImageTest")
      .Input("Input")
      .Output("B2IOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  const half *h_data = reinterpret_cast<const half *>(input_data);

  net.AddInputFromArray<D, half>("Input", input_shape,
                                 std::vector<half>(h_data, h_data + 2));

  // Run
  net.RunOp(D);

  OpDefBuilder("ImageToBuffer", "ImageToBufferTest")
      .Input("B2IOutput")
      .Output("I2BOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

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
      kernels::ARGUMENT, {2}, input_data);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
