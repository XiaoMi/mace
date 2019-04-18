// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#include <gtest/gtest.h>

#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/q8/gemv.h"
#include "mace/ops/ref/gemv.h"
#include "mace/ops/testing/test_utils.h"

namespace mace {
namespace ops {
namespace test {

void TestGemvInt32(const index_t batch,
                   const index_t height,
                   const index_t width,
                   const bool lhs_batched,
                   const bool rhs_batched) {
  Tensor lhs(GetCPUAllocator(), DataType::DT_UINT8);
  Tensor rhs(GetCPUAllocator(), DataType::DT_UINT8);
  Tensor bias(GetCPUAllocator(), DataType::DT_INT32);
  Tensor output(GetCPUAllocator(), DataType::DT_INT32);
  lhs.SetScale(0.5);
  rhs.SetScale(0.3);
  lhs.SetZeroPoint(0);
  rhs.SetZeroPoint(0);
  lhs.Resize({lhs_batched ? batch : 1, height, width});
  rhs.Resize({rhs_batched ? batch : 1, batch, width});
  bias.Resize({height});
  output.Resize({batch, height});
  {
    Tensor::MappingGuard lhs_guard(&lhs);
    Tensor::MappingGuard rhs_guard(&rhs);
    Tensor::MappingGuard bias_guard(&bias);
    uint8_t *lhs_data = lhs.mutable_data<uint8_t>();
    uint8_t *rhs_data = rhs.mutable_data<uint8_t>();
    int32_t *bias_data = bias.mutable_data<int32_t>();
    GenerateRandomIntTypeData<uint8_t>(lhs.shape(), lhs_data);
    GenerateRandomIntTypeData<uint8_t>(rhs.shape(), rhs_data);
    GenerateRandomIntTypeData<int32_t>(bias.shape(), bias_data);
  }

  utils::ThreadPool thread_pool(1, AFFINITY_NONE);
  thread_pool.Init();
  CPUDevice cpu_device(1, AFFINITY_NONE, &thread_pool);
  OpContext context(nullptr, &cpu_device);
  mace::ops::arm::q8::Gemv<int32_t> gemv;
  gemv.Compute(&context,
               &lhs,
               &rhs,
               &bias,
               batch,
               height,
               width,
               lhs_batched,
               rhs_batched,
               &output);

  Tensor expected_output(GetCPUAllocator(), DataType::DT_INT32);
  expected_output.Resize({batch, height});
  mace::ops::ref::Gemv<int32_t> gemv_ref;
  gemv_ref.Compute(nullptr,
                   &lhs,
                   &rhs,
                   &bias,
                   batch,
                   height,
                   width,
                   lhs_batched,
                   rhs_batched,
                   &expected_output);

  Tensor::MappingGuard output_guard(&output);
  Tensor::MappingGuard expected_guard(&expected_output);
  const int32_t *output_data = output.data<int32_t>();
  const int32_t *expected_data = expected_output.data<int32_t>();

  for (index_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

void TestGemvUint8(const index_t batch,
                   const index_t height,
                   const index_t width,
                   const bool lhs_batched,
                   const bool rhs_batched) {
  Tensor lhs(GetCPUAllocator(), DataType::DT_UINT8);
  Tensor rhs(GetCPUAllocator(), DataType::DT_UINT8);
  Tensor bias(GetCPUAllocator(), DataType::DT_INT32);
  Tensor output(GetCPUAllocator(), DataType::DT_UINT8);
  lhs.SetScale(0.5);
  rhs.SetScale(0.3);
  output.SetScale(0.6);
  lhs.SetZeroPoint(23);
  rhs.SetZeroPoint(45);
  output.SetZeroPoint(57);
  lhs.Resize({batch, height, width});
  rhs.Resize({batch, width});
  bias.Resize({height});
  output.Resize({batch, height});
  {
    Tensor::MappingGuard lhs_guard(&lhs);
    Tensor::MappingGuard rhs_guard(&rhs);
    Tensor::MappingGuard bias_guard(&bias);

    uint8_t *lhs_data = lhs.mutable_data<uint8_t>();
    uint8_t *rhs_data = rhs.mutable_data<uint8_t>();
    int32_t *bias_data = bias.mutable_data<int32_t>();
    GenerateRandomIntTypeData<uint8_t>(lhs.shape(), lhs_data);
    GenerateRandomIntTypeData<uint8_t>(rhs.shape(), rhs_data);
    GenerateRandomIntTypeData<int32_t>(bias.shape(), bias_data);
  }

  utils::ThreadPool thread_pool(1, AFFINITY_NONE);
  thread_pool.Init();
  CPUDevice cpu_device(1, AFFINITY_NONE, &thread_pool);
  OpContext context(nullptr, &cpu_device);
  mace::ops::arm::q8::Gemv<uint8_t> gemv;
  gemv.Compute(&context,
               &lhs,
               &rhs,
               &bias,
               batch,
               height,
               width,
               lhs_batched,
               rhs_batched,
               &output);

  Tensor expected_output(GetCPUAllocator(), DataType::DT_INT32);
  expected_output.SetScale(0.6);
  expected_output.SetZeroPoint(57);
  expected_output.Resize({batch, height});
  mace::ops::ref::Gemv<uint8_t> gemv_ref;
  gemv_ref.Compute(nullptr,
                   &lhs,
                   &rhs,
                   &bias,
                   batch,
                   height,
                   width,
                   lhs_batched,
                   rhs_batched,
                   &expected_output);

  Tensor::MappingGuard output_guard(&output);
  Tensor::MappingGuard expected_guard(&expected_output);
  const uint8_t *output_data = output.data<uint8_t>();
  const uint8_t *expected_data = expected_output.data<uint8_t>();

  for (index_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

TEST(ArmGemv, TestGemvInt32) {
  TestGemvInt32(1, 16, 4, true, true);
  TestGemvInt32(1, 16, 256, true, true);
  TestGemvInt32(2, 16, 256, true, true);
  TestGemvInt32(3, 63, 257, true, true);

  TestGemvInt32(2, 16, 256, false, true);
  TestGemvInt32(3, 63, 257, false, true);
  TestGemvInt32(2, 16, 256, true, false);
  TestGemvInt32(3, 63, 257, true, false);
}

TEST(ArmGemv, TestGemvUint8) {
  TestGemvUint8(1, 16, 4, true, true);
  TestGemvUint8(1, 16, 256, true, true);
  TestGemvUint8(2, 16, 256, true, true);
  TestGemvUint8(3, 63, 257, true, true);

  TestGemvUint8(2, 16, 256, false, true);
  TestGemvUint8(3, 63, 257, false, true);
  TestGemvUint8(2, 16, 256, true, false);
  TestGemvUint8(3, 63, 257, true, false);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
