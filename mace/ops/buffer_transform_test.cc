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

#include <cstring>

#include "gtest/gtest.h"
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
void TestBidirectionTransform(const int type,
                              const std::vector<index_t> &input_shape) {
  OpsTestNet net;
  OpDefBuilder("BufferTransform", "BufferTransformTest")
      .Input("Input")
      .Output("TransformedOutput")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<DstType>::value)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::GPU, OrgType>("Input", input_shape);

  // Run
  net.RunOp(DeviceType::GPU);

  OpDefBuilder("BufferInverseTransform", "BufferInverseTransformTest")
      .Input("TransformedOutput")
      .Output("Output")
      .AddIntArg("buffer_type", type)
      .AddIntArg("T", DataTypeToEnum<OrgType>::value)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

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
  TestBidirectionTransform<float, half>(kernels::BufferType::IN_OUT_CHANNEL,
                                        {1, 2, 3, 4});
}

TEST_F(BufferTransformTest, HalfToHalf) {
  TestBidirectionTransform<half, half>(kernels::BufferType::IN_OUT_CHANNEL,
                                       {1, 2, 3, 4});
}

namespace {
template <typename T>
void TestArgumentTransform(const index_t input_size) {
  OpsTestNet net;
  OpDefBuilder("BufferTransform", "BufferTransformTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("buffer_type", kernels::BufferType::ARGUMENT)
      .AddIntArg("T", DataTypeToEnum<T>::value)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::GPU, T>("Input", {input_size});

  // Run
  net.RunOp(DeviceType::GPU);

  auto output_tensor = net.GetOutput("Output");
  index_t expected_size = RoundUp<index_t>(input_size, 4);
  EXPECT_EQ(expected_size, output_tensor->buffer_shape()[0]);

  // Check
  ExpectTensorNear<T>(*net.GetTensor("Input"), *output_tensor,
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
