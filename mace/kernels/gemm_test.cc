//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include <gtest/gtest.h>
#include <random>

#include "mace/kernels/gemm.h"
#include "mace/core/types.h"

namespace mace {

TEST(GEMMTest, gemm) {
  index_t N = 17;
  index_t M = 33;
  index_t K = 64;
  float *A = new float[N * K];
  float *B = new float[K * M];
  float *C = new float[N * M];
  float *C_ref = new float[N * M];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  std::generate(A, A + N * K,
                [&gen, &nd] {
                  return nd(gen);
                });
  std::generate(B, B + K * M,
                [&gen, &nd] {
                  return nd(gen);
                });
  kernels::Gemm(A, B, N, K, M, C);
  kernels::GemmRef(A, B, N, K, M, C_ref);

  for (int i = 0; i < N * M; ++i) {
    EXPECT_NEAR(C_ref[i], C[i], 0.1);
  }

  delete[]A;
  delete[]B;
  delete[]C;
}

}  // namespace mace
