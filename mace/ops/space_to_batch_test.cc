//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

template <DeviceType D>
void RunSpaceToBatch(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const std::vector<index_t> &block_shape_shape,
                     const std::vector<int> &block_shape_data,
                     const std::vector<index_t> &padding_shape,
                     const std::vector<int> &padding_data,
                     const Tensor *expected) {
  OpsTestNet net;
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("Input")
      .Input("BlockShape")
      .Input("Padding")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", input_shape, input_data);
  net.AddInputFromArray<D, int>(
      "BlockShape", block_shape_shape, block_shape_data);
  net.AddInputFromArray<D, int>("Padding", padding_shape, padding_data);

  // Run
  net.RunOp(D);

  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-8);

}

template <DeviceType D>
void RunBatchToSpace(const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const std::vector<index_t> &block_shape_shape,
                     const std::vector<int> &block_shape_data,
                     const std::vector<index_t> &crops_shape,
                     const std::vector<int> &crops_data,
                     const Tensor *expected) {
  OpsTestNet net;
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("Input")
      .Input("BlockShape")
      .Input("Crops")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", input_shape, input_data);
  net.AddInputFromArray<D, int>(
      "BlockShape", block_shape_shape, block_shape_data);
  net.AddInputFromArray<D, int>("Crops", crops_shape, crops_data);

  // Run
  net.RunOp(D);

  // Check
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-8);
}

template <typename T>
void TestBidirectionTransform(const std::vector<index_t> &space_shape,
                              const std::vector<float> &space_data,
                              const std::vector<index_t> &block_shape,
                              const std::vector<int> &block_data,
                              const std::vector<index_t> &padding_shape,
                              const std::vector<int> &padding_data,
                              const std::vector<index_t> &batch_shape,
                              const std::vector<float> &batch_data) {

  auto space_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                                    DataTypeToEnum<T>::v()));
  space_tensor->Resize(space_shape);
  {
    Tensor::MappingGuard space_mapper(space_tensor.get());
    T *space_ptr = space_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(space_tensor->size()) == space_data.size())
      << "Space tensor size:" << space_tensor->size()
      << ", space data size:" << space_data.size();
    memcpy(space_ptr, space_data.data(), space_data.size() * sizeof(T));
  }

  auto batch_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
                                                    DataTypeToEnum<T>::v()));
  batch_tensor->Resize(batch_shape);
  {
    Tensor::MappingGuard batch_mapper(batch_tensor.get());
    T *batch_ptr = batch_tensor->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(batch_tensor->size()) == batch_data.size());
    memcpy(batch_ptr, batch_data.data(), batch_data.size() * sizeof(T));
  }

  RunSpaceToBatch<DeviceType::OPENCL>(space_shape, space_data,
                                      block_shape, block_data,
                                      padding_shape, padding_data,
                                      batch_tensor.get());

  RunBatchToSpace<DeviceType::OPENCL>(batch_shape, batch_data,
                                      block_shape, block_data,
                                      padding_shape, padding_data,
                                      space_tensor.get());
}

TEST(SpaceToBatchTest, SmallData) {
  TestBidirectionTransform<float>({1, 1, 2, 2},
                                  {1,2,3,4},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {0, 0, 0, 0},
                                  {4,1,1,1},
                                  {1,2,3,4}
  );
}

TEST(SpaceToBatchTest, SmallDataWithOnePadding) {
  TestBidirectionTransform<float>({1, 1, 2, 2},
                                  {1,2,3,4},
                                  {2},
                                  {3, 3},
                                  {2, 2},
                                  {1, 0, 1, 0},
                                  {9,1,1,1},
                                  {0,0,0,0,1,2,0,3,4}
  );
}

TEST(SpaceToBatchTest, SmallDataWithTwoPadding) {
  TestBidirectionTransform<float>({1, 1, 2, 2},
                                  {1,2,3,4},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {1, 1, 1, 1},
                                  {4,1,2,2},
                                  {0,0,0,4,0,0,3,0,0,2,0,0,1,0,0,0}
  );
}

TEST(SpaceToBatchTest, MultiChannelData) {
  TestBidirectionTransform<float>({1, 3, 2, 2},
                                  {1,2,3,4,5,6,7,8,9,10,11,12},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {0, 0, 0, 0},
                                  {4,3,1,1},
                                  {1,5,9,2,6,10,3,7,11,4,8,12}
                                  );
}

TEST(SpaceToBatchTest, LargerMultiChannelData) {
  TestBidirectionTransform<float>({1, 1, 4, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {0, 0, 0, 0},
                                  {4,1,2,2},
                                  {1,3,9,11,2,4,10,12,5,7,13,15,6,8,14,16}
  );
}

TEST(SpaceToBatchTest, MultiBatchData) {
  TestBidirectionTransform<float>({2, 1, 2, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {0, 0, 0, 0},
                                  {8,1,1,2},
                                  {1,3,2,4,5,7,6,8,9,11,10,12,13,15,14,16}
  );
}

TEST(SpaceToBatchTest, MultiBatchAndChannelData) {
  TestBidirectionTransform<float>({2, 2, 2, 4},
                                  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                                   17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32},
                                  {2},
                                  {2, 2},
                                  {2, 2},
                                  {0, 0, 0, 0},
                                  {8,2,1,2},
                                  {1,3,9,11,2,4,10,12,5,7,13,15,6,8,14,16,
                                  17,19,25,27,18,20,26,28,21,23,29,31,22,24,30,32}
  );
}

