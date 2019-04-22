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

#include <cstring>

#include "gtest/gtest.h"
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class BufferTransformTest : public OpsTestBase {
 protected:
  virtual void SetUp() {
    OpTestContext::Get()->SetOCLBufferTestFlag();
  }
};

namespace {
template <typename OrgType, typename DstType>
void TestBidirectionTransform(const OpenCLBufferType type,
                              const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  net.AddRandomInput<DeviceType::GPU, OrgType>("Input", input_shape);
  Tensor *bt_output = net.ws()->CreateTensor(
      "BtOutput", context.device()->allocator(),
      DataTypeToEnum<DstType>::value);

  OpenCLBufferTransformer<DstType>(MemoryType::GPU_BUFFER,
                                   MemoryType::GPU_BUFFER)
      .Transform(&context, net.ws()->GetTensor("Input"),
                 type, MemoryType::GPU_BUFFER, 0, bt_output);

  // Inverse Transform
  Tensor *output = net.ws()->CreateTensor(
      "Output", context.device()->allocator(),
      DataTypeToEnum<OrgType>::value);
  OpenCLBufferTransformer<OrgType>(MemoryType::GPU_BUFFER,
                                   MemoryType::GPU_BUFFER)
      .Transform(&context, bt_output,
                 type, MemoryType::GPU_BUFFER, 0, output);

  if (DataTypeToEnum<OrgType>::value == DataTypeToEnum<DstType>::value) {
    EXPECT_EQ(net.GetOutput("Input")->UnderlyingBuffer(),
              net.GetOutput("Output")->UnderlyingBuffer());
  } else {
    // Check
    ExpectTensorNear<OrgType>(*net.GetOutput("Input"),
                              *net.GetOutput("Output"),
                              1e-3, 1e-4);
  }
}
}  // namespace

TEST_F(BufferTransformTest, FloatToHalf) {
  TestBidirectionTransform<float, half>(OpenCLBufferType::IN_OUT_CHANNEL,
                                        {1, 2, 3, 4});
}

namespace {
template <typename T>
void TestArgumentTransform(const index_t input_size) {
  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  net.AddRandomInput<DeviceType::GPU, T>("Input", {input_size});

  // Run
  Tensor *output = net.ws()->CreateTensor(
      "Output", context.device()->allocator(),
      DataTypeToEnum<T>::value);
  OpenCLBufferTransformer<T>(MemoryType::GPU_BUFFER,
                             MemoryType::GPU_BUFFER)
      .Transform(&context, net.ws()->GetTensor("Input"),
                 OpenCLBufferType::ARGUMENT, MemoryType::GPU_BUFFER,
                 0, output);

  index_t expected_size = RoundUp<index_t>(input_size, 4);
  EXPECT_EQ(expected_size, output->buffer_shape()[0]);

  // Check
  ExpectTensorNear<T>(*net.GetTensor("Input"), *output,
                      1e-3, 1e-4);
}
}  // namespace

TEST_F(BufferTransformTest, Argument) {
  TestArgumentTransform<half>(30);
  TestArgumentTransform<half>(32);
}

}  // namespace test
}  // namespace ops
}  // namespace mace


#endif  // MACE_ENABLE_OPENCL
