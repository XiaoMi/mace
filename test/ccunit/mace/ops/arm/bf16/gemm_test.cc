// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/delegator/gemm.h"

#include <gtest/gtest.h>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/testing/test_utils.h"

namespace mace {
namespace ops {
namespace test {

void TestGemmBFloat16(const index_t batch,
                      const index_t rows,
                      const index_t cols,
                      const index_t depth,
                      const MatrixMajor lhs_major,
                      const MatrixMajor rhs_major,
                      const MatrixMajor output_major,
                      const bool lhs_batched,
                      const bool rhs_batched) {
  Tensor lhs(GetCPUAllocator(), DT_BFLOAT16);
  Tensor rhs(GetCPUAllocator(), DT_BFLOAT16);
  Tensor output(GetCPUAllocator(), DT_BFLOAT16);
  lhs.Resize({lhs_batched ? batch : 1, rows, depth});
  rhs.Resize({rhs_batched ? batch : 1, depth, cols});
  output.Resize({batch, rows, cols});
  {
    Tensor::MappingGuard lhs_guard(&lhs);
    Tensor::MappingGuard rhs_guard(&rhs);
    auto lhs_data = lhs.mutable_data<BFloat16>();
    auto rhs_data = rhs.mutable_data<BFloat16>();
    auto output_data = output.mutable_data<BFloat16>();
    GenerateRandomRealTypeData<BFloat16>(lhs.shape(), lhs_data);
    GenerateRandomRealTypeData<BFloat16>(rhs.shape(), rhs_data);
    GenerateRandomRealTypeData<BFloat16>(output.shape(), output_data);
  }

  utils::ThreadPool thread_pool(1, AFFINITY_NONE);
  thread_pool.Init();
  CPUDevice cpu_device(1, AFFINITY_NONE, &thread_pool);
  OpsTestNet net;
  OpContext context(net.ws(), &cpu_device);
  std::unique_ptr<delegator::Gemm> gemm = delegator::Gemm::Create(
      context.workspace(),
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, BFloat16, ImplType::NEON),
      delegator::GemmParam());
  gemm->Compute(&context, &lhs, &rhs, batch, rows, cols, depth, lhs_major,
                rhs_major, output_major, lhs_batched, rhs_batched, &output);

  Tensor expected_output(GetCPUAllocator(), DataType::DT_BFLOAT16);
  expected_output.Resize({batch, rows, cols});
  std::unique_ptr<delegator::Gemm> gemm_ref = delegator::Gemm::Create(
      context.workspace(),
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, BFloat16, ImplType::REF),
      delegator::GemmParam());
  gemm_ref->Compute(&context, &lhs, &rhs, batch, rows, cols, depth, lhs_major,
                    rhs_major, output_major, lhs_batched, rhs_batched,
                    &expected_output);

  ExpectTensorSimilar<BFloat16>(expected_output, output, 1e-4);
}

TEST(ArmGemm, TestGemmBF16) {
  TestGemmBFloat16(1, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, RowMajor, RowMajor, ColMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, RowMajor, ColMajor, RowMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, RowMajor, ColMajor, ColMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, ColMajor, RowMajor, RowMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, ColMajor, RowMajor, ColMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, ColMajor, ColMajor, RowMajor, true, true);
  TestGemmBFloat16(1, 47, 69, 37, ColMajor, ColMajor, ColMajor, true, true);

  TestGemmBFloat16(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, RowMajor, RowMajor, ColMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, RowMajor, ColMajor, RowMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, RowMajor, ColMajor, ColMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, ColMajor, RowMajor, RowMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, ColMajor, RowMajor, ColMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, ColMajor, ColMajor, RowMajor, true, true);
  TestGemmBFloat16(3, 47, 69, 37, ColMajor, ColMajor, ColMajor, true, true);

  TestGemmBFloat16(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, true, false);
  TestGemmBFloat16(3, 47, 69, 37, RowMajor, RowMajor, RowMajor, false, true);

  TestGemmBFloat16(16, 31, 61, 67, RowMajor, ColMajor, RowMajor, true, true);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
