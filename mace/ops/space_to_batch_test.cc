//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/opencl/space_to_batch.h"
#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

template <typename T>
void TestBidirectionTransform(const std::vector<index_t> &space_shape,
                              const std::vector<float> &space,
                              const int block_height,
                              const int block_width,
                              const std::vector<index_t> &batch_shape,
                              const std::vector<float> &batch) {

  auto space_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                                    DataTypeToEnum<T>::v()));
  space_tensor->Resize(space_shape);
  {
    Tensor::MappingGuard space_mapper(space_tensor.get());
    T *space_data = space_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(space_tensor->size()) == space.size())
      << "Space tensor size:" << space_tensor->size()
      << ", space data size:" << space.size();
    memcpy(space_data, space.data(), space.size() * sizeof(T));
  }

  auto batch_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                                    DataTypeToEnum<T>::v()));
  batch_tensor->Resize(batch_shape);
  {
    Tensor::MappingGuard batch_mapper(batch_tensor.get());
    T *batch_data = batch_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(batch_tensor->size()) == batch.size());
    memcpy(batch_data, batch.data(), batch.size() * sizeof(T));
  }

  auto inner_batch_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                               DataTypeToEnum<T>::v()));
  inner_batch_tensor->Resize(batch_shape);
  kernels::SpaceToBatch(space_tensor.get(), block_height, block_width,
                        inner_batch_tensor.get(), nullptr, nullptr);
  ExpectTensorNear<float>(*batch_tensor, *inner_batch_tensor, 1e-8);
  auto inner_space_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                                          DataTypeToEnum<T>::v()));
  inner_space_tensor->Resize(space_shape);
  kernels::SpaceToBatch<true>(inner_space_tensor.get(), block_height, block_width,
                              batch_tensor.get(), nullptr, nullptr);
  ExpectTensorNear<float>(*space_tensor, *inner_space_tensor, 1e-8);
}

TEST(SpaceToBatchTest, NoTransform) {
  TestBidirectionTransform<float>({1, 1, 2, 2},
                                  {1,2,3,4},
                                  1, 1,
                                  {1,1,2,2},
                                  {1,2,3,4});
}

TEST(SpaceToBatchTest, SmallData) {
  TestBidirectionTransform<float>({1, 1, 2, 2},
                                  {1,2,3,4},
                                  2, 2,
                                  {4,1,1,1},
                                  {1,2,3,4});
}

TEST(SpaceToBatchTest, MultiChannelData) {
  TestBidirectionTransform<float>({1, 3, 2, 2},
                                  {1,2,3,4,5,6,7,8,9,10,11,12},
                                  2, 2,
                                  {4,3,1,1},
                                  {1,5,9,2,6,10,3,7,11,4,8,12}
                                  );
}

TEST(SpaceToBatchTest, LargerMultiChannelData) {
  TestBidirectionTransform<float>({1, 1, 4, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
                                  2, 2,
                                  {4,1,2,2},
                                  {1,3,9,11,2,4,10,12,5,7,13,15,6,8,14,16}
  );
}

TEST(SpaceToBatchTest, MultiBatchData) {
  TestBidirectionTransform<float>({2, 1, 2, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
                                  2, 2,
                                  {8,1,1,2},
                                  {1,3,2,4,5,7,6,8,9,11,10,12,13,15,14,16}
  );
}

TEST(SpaceToBatchTest, MultiBatchAndChannelData) {
  TestBidirectionTransform<float>({2, 2, 2, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                                   17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32},
                                  2, 2,
                                  {8,2,1,2},
                                  {1,3,9,11,2,4,10,12,5,7,13,15,6,8,14,16,
                                  17,19,25,27,18,20,26,28,21,23,29,31,22,24,30,32}
  );
}

