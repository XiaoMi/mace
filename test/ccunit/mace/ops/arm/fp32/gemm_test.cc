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
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/ref/gemm.h"
#include "mace/ops/testing/test_utils.h"

namespace mace {
namespace ops {
namespace test {

void TestGemmFloat32(const index_t batch,
                     const index_t rows,
                     const index_t cols,
                     const index_t depth,
                     const MatrixMajor lhs_major,
                     const MatrixMajor rhs_major,
                     const MatrixMajor output_major,
                     const bool lhs_batched,
                     const bool rhs_batched) {
  Tensor lhs(GetCPUAllocator(), DataType::DT_FLOAT);
  Tensor rhs(GetCPUAllocator(), DataType::DT_FLOAT);
  Tensor output(GetCPUAllocator(), DataType::DT_FLOAT);
  lhs.Resize({lhs_batched ? batch : 1, rows, depth});
  rhs.Resize({rhs_batched ? batch : 1, depth, cols});
  output.Resize({batch, rows, cols});
  {
    Tensor::MappingGuard lhs_guard(&lhs);
    Tensor::MappingGuard rhs_guard(&rhs);
    float *lhs_data = lhs.mutable_data<float>();
    float *rhs_data = rhs.mutable_data<float>();
    float *output_data = output.mutable_data<float>();
    GenerateRandomRealTypeData<float>(lhs.shape(), lhs_data);
    GenerateRandomRealTypeData<float>(rhs.shape(), rhs_data);
    GenerateRandomRealTypeData<float>(output.shape(), output_data);
  }
  ::mace::ops::arm::fp32::Gemm gemm;
  utils::ThreadPool thread_pool(1, AFFINITY_NONE);
  thread_pool.Init();
  CPUDevice cpu_device(1, AFFINITY_NONE, &thread_pool);
  OpContext context(nullptr, &cpu_device);
  gemm.Compute(&context,
               &lhs,
               &rhs,
               batch,
               rows,
               cols,
               depth,
               lhs_major,
               rhs_major,
               output_major,
               lhs_batched,
               rhs_batched,
               &output);

  Tensor expected_output(GetCPUAllocator(), DataType::DT_FLOAT);
  expected_output.Resize({batch, rows, cols});
  ::mace::ops::ref::Gemm<float> gemm_ref;
  gemm_ref.Compute(nullptr,
                   &lhs,
                   &rhs,
                   batch,
                   rows,
                   cols,
                   depth,
                   lhs_major,
                   rhs_major,
                   output_major,
                   lhs_batched,
                   rhs_batched,
                   &expected_output);

  ExpectTensorNear<float>(expected_output, output);
}

TEST(ArmGemm, TestGemmFloat32) {
  TestGemmFloat32(1, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, RowMajor, RowMajor, ColMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, RowMajor, ColMajor, RowMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, RowMajor, ColMajor, ColMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, ColMajor, RowMajor, RowMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, ColMajor, RowMajor, ColMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, ColMajor, ColMajor, RowMajor, true, true);
  TestGemmFloat32(1, 47, 69, 37, ColMajor, ColMajor, ColMajor, true, true);

  TestGemmFloat32(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, RowMajor, RowMajor, ColMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, RowMajor, ColMajor, RowMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, RowMajor, ColMajor, ColMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, ColMajor, RowMajor, RowMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, ColMajor, RowMajor, ColMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, ColMajor, ColMajor, RowMajor, true, true);
  TestGemmFloat32(3, 47, 69, 37, ColMajor, ColMajor, ColMajor, true, true);

  TestGemmFloat32(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, false);
  TestGemmFloat32(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, false, true);

  TestGemmFloat32(16, 31, 61, 67, RowMajor, ColMajor, RowMajor, true, true);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
