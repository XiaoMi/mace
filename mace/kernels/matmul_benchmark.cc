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

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <vector>

#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/gemm.h"

namespace mace {
namespace kernels {
namespace test {

// Test the speed of different access order of a NHWC buffer

namespace {

// Matmul with (m, k) x (k, n)
void MatmulBenchmark_Mace(int iters, int m, int k, int n) {
  mace::testing::StopTiming();
  std::vector<float> lhs(m * k);
  std::vector<float> rhs(k * n);
  std::vector<float> result(m * n);
  // warm up
  Gemm(lhs.data(), rhs.data(), 1, m, k, n, result.data());
  mace::testing::StartTiming();
  while (iters--) {
    Gemm(lhs.data(), rhs.data(), 1, m, k, n, result.data());
  }
}

void MatmulBenchmark_Eigen(int iters, int m, int k, int n) {
  mace::testing::StopTiming();
  Eigen::MatrixXd lhs = Eigen::MatrixXd::Random(m, k);
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(k, n);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m, n);
  // warm up
  result = lhs * rhs;
  mace::testing::StartTiming();
  while (iters--) {
    result = lhs * rhs;
  }
}

}  // namespace

#define MACE_BM_MATMUL_FUNC(M, K, N, FUNC)                         \
  static void MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC(int iters) { \
    const int64_t macc = static_cast<int64_t>(iters) * M * K * N;  \
    const int64_t tot = static_cast<int64_t>(iters) * (M + N) * K; \
    mace::testing::MaccProcessed(macc);                            \
    mace::testing::BytesProcessed(tot * sizeof(float));            \
    MatmulBenchmark_##FUNC(iters, M, K, N);                        \
  }                                                                \
  MACE_BENCHMARK(MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC)

#define MACE_BM_MATMUL(M, K, N)        \
  MACE_BM_MATMUL_FUNC(M, K, N, Mace);  \
  MACE_BM_MATMUL_FUNC(M, K, N, Eigen);

// Embedding size 384
MACE_BM_MATMUL(7, 384, 384);
MACE_BM_MATMUL(7, 384, 1536);
MACE_BM_MATMUL(7, 1536, 384);

MACE_BM_MATMUL(15, 384, 384);
MACE_BM_MATMUL(15, 384, 1536);
MACE_BM_MATMUL(15, 1536, 384);

MACE_BM_MATMUL(1, 384, 384);
MACE_BM_MATMUL(1, 384, 1536);
MACE_BM_MATMUL(1, 1536, 384);
MACE_BM_MATMUL(1, 384, 44678);

// Embedding size 128
MACE_BM_MATMUL(1, 128, 1536);
MACE_BM_MATMUL(1, 128, 44678);

}  // namespace test
}  // namespace kernels
}  // namespace mace
