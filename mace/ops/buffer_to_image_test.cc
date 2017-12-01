//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"
#include "mace/kernels/opencl/helper.h"

using namespace mace;

template<DeviceType D, typename T>
void TestBidirectionTransform(const int type, const std::vector<index_t> &input_shape) {
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
  ExpectTensorNear<T>(*net.GetOutput("Input"), *net.GetOutput("I2BOutput"), 1e-5);
}

TEST(BufferToImageTest, ArgSmall) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::ARGUMENT, {1});
}

TEST(BufferToImageTest, ArgHalfSmall) {
  TestBidirectionTransform<DeviceType::OPENCL, half>(kernels::ARGUMENT, {1});
}

TEST(BufferToImageTest, ArgMedia) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::ARGUMENT, {11});
}

TEST(BufferToImageTest, ArgLarge) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::ARGUMENT, {256});
}

TEST(BufferToImageTest, InputSmallSingleChannel) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::IN_OUT, {1, 2, 3, 1});
}

TEST(BufferToImageTest, InputSmallMultipleChannel) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::IN_OUT, {1, 2, 3, 3});
}

TEST(BufferToImageTest, InputSmallMultipleBatchAndChannel) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::IN_OUT, {3, 2, 3, 3});
}

TEST(BufferToImageTest, InputMedia) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::IN_OUT, {3, 13, 17, 128});
}

TEST(BufferToImageTest, InputLarge) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::IN_OUT, {3, 64, 64, 256});
}

TEST(BufferToImageTest, Filter1x1Small) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {1, 1, 3, 5});
}

TEST(BufferToImageTest, Filter1x1Media) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {1, 1, 13, 17});
}

TEST(BufferToImageTest, Filter1x1Large) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {1, 1, 128, 512});
}

TEST(BufferToImageTest, Filter3x3Small) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {3, 3, 3, 5});
}

TEST(BufferToImageTest, Filter3x3Meida) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {3, 3, 13, 17});
}

TEST(BufferToImageTest, Filter3x3Large) {
  TestBidirectionTransform<DeviceType::OPENCL, float>(kernels::FILTER, {3, 3, 128, 256});
}
