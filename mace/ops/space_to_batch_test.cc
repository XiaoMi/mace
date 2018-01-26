//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

template <DeviceType D>
void RunSpaceToBatch(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const std::vector<int> &block_shape_data,
                     const std::vector<int> &padding_data,
                     const Tensor *expected) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("paddings", padding_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  ImageToBuffer<D, float>(net, "OutputImage", "Output",
                          kernels::BufferType::IN_OUT_CHANNEL);
  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-8);
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

  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("crops", crops_data)
      .AddIntsArg("block_shape", block_shape_data)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  ImageToBuffer<D, float>(net, "OutputImage", "Output",
                          kernels::BufferType::IN_OUT_CHANNEL);
  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-8);
}

template <typename T>
void TestBidirectionalTransform(const std::vector<index_t> &space_shape,
                                const std::vector<float> &space_data,
                                const std::vector<int> &block_data,
                                const std::vector<int> &padding_data,
                                const std::vector<index_t> &batch_shape,
                                const std::vector<float> &batch_data) {
  auto space_tensor = unique_ptr<Tensor>(new Tensor(
      GetDeviceAllocator(DeviceType::OPENCL), DataTypeToEnum<T>::v()));
  space_tensor->Resize(space_shape);
  {
    Tensor::MappingGuard space_mapper(space_tensor.get());
    T *space_ptr = space_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(space_tensor->size()) == space_data.size())
        << "Space tensor size:" << space_tensor->size()
        << ", space data size:" << space_data.size();
    memcpy(space_ptr, space_data.data(), space_data.size() * sizeof(T));
  }

  auto batch_tensor = unique_ptr<Tensor>(new Tensor(
      GetDeviceAllocator(DeviceType::OPENCL), DataTypeToEnum<T>::v()));
  batch_tensor->Resize(batch_shape);
  {
    Tensor::MappingGuard batch_mapper(batch_tensor.get());
    T *batch_ptr = batch_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(batch_tensor->size()) == batch_data.size());
    memcpy(batch_ptr, batch_data.data(), batch_data.size() * sizeof(T));
  }

  RunSpaceToBatch<DeviceType::OPENCL>(space_shape, space_data, block_data,
                                      padding_data, batch_tensor.get());

  RunBatchToSpace<DeviceType::OPENCL>(batch_shape, batch_data, block_data,
                                      padding_data, space_tensor.get());
}

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
      {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16});
}

TEST(SpaceToBatchTest, MultiBatchAndChannelData) {
  TestBidirectionalTransform<float>(
      {2, 2, 4, 2},
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
      {2, 2}, {0, 0, 0, 0}, {8, 1, 2, 2},
      {1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12, 15, 16,
       17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32});
}

// TEST(SpaceTobatchTest, CompareTF) {
//
//  const std::string space_file = "/data/local/tmp/test/input";
//  const std::string batch_file = "/data/local/tmp/test/output";
//  const std::vector<index_t> space_shape = {1, 256, 256, 32};
//  const int space_size = std::accumulate(space_shape.begin(),
//  space_shape.end(), 1, std::multiplies<int>());
//  const std::vector<index_t> batch_shape = {4, 130, 130, 32};
//  const int batch_size = std::accumulate(batch_shape.begin(),
//  batch_shape.end(), 1, std::multiplies<int>());
//
//  auto space_tensor = unique_ptr<Tensor>(new
//  Tensor(GetDeviceAllocator(DeviceType::OPENCL),
//                                                    DataTypeToEnum<float>::v()));
//  space_tensor->Resize(space_shape);
//  std::vector<float> space_data(space_size, 0.0);
//  std::ifstream in_file(space_file, std::ios::in | std::ios::binary);
//  if (in_file.is_open()) {
//    in_file.read(reinterpret_cast<char *>(space_data.data()),
//                 space_size * sizeof(float));
//    in_file.close();
//    Tensor::MappingGuard space_mapper(space_tensor.get());
//    float *space_ptr = space_tensor->mutable_data<float>();
//    MACE_CHECK(static_cast<size_t>(space_tensor->size()) == space_data.size())
//      << "Space tensor size:" << space_tensor->size()
//      << ", space data size:" << space_data.size();
//    memcpy(space_ptr, space_data.data(), space_data.size() * sizeof(float));
//  } else {
//    VLOG(0) << "open space file failed";
//  }
//
//  auto batch_tensor = unique_ptr<Tensor>(new
//  Tensor(GetDeviceAllocator(DeviceType::OPENCL),
//                                                    DataTypeToEnum<float>::v()));
//  std::vector<float> batch_data(batch_size, 0.0);
//  batch_tensor->Resize(batch_shape);
//  {
//    std::ifstream in_file(batch_file, std::ios::in | std::ios::binary);
//    if (in_file.is_open()) {
//      in_file.read(reinterpret_cast<char *>(batch_data.data()),
//                    batch_size * sizeof(float));
//      in_file.close();
//    } else {
//      VLOG(0) << "open batch file failed";
//    }
//    Tensor::MappingGuard batch_mapper(batch_tensor.get());
//    float *batch_ptr = batch_tensor->mutable_data<float>();
//    MACE_CHECK(static_cast<size_t>(batch_tensor->size()) ==
//    batch_data.size());
//    memcpy(batch_ptr, batch_data.data(), batch_data.size() * sizeof(float));
//  }
//
//  RunSpaceToBatch<DeviceType::OPENCL>(space_shape, space_data,
//                                      {2, 2},
//                                      {2, 2, 2, 2},
//                                      batch_tensor.get());
//
//  RunBatchToSpace<DeviceType::OPENCL>(batch_shape, batch_data,
//                                      {2, 2},
//                                      {2, 2, 2, 2},
//                                      space_tensor.get());
//}
