//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include <gtest/gtest.h>
#include <random>
#include <memory>

#include "mace/kernels/gemm.h"
#include "mace/core/types.h"

namespace mace {

TEST(GEMMTest, gemm) {
  index_t N = 17;
  index_t M = 33;
  index_t K = 64;
  std::unique_ptr<float[]> A(new float[N * K]);
  std::unique_ptr<float[]> B(new float[K * M]);
  std::unique_ptr<float[]> C(new float[N * M]);
  std::unique_ptr<float[]> C_ref(new float[N * M]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  std::generate(A.get(), A.get() + N * K,
                [&gen, &nd] {
                  return nd(gen);
                });
  std::generate(B.get(), B.get() + K * M,
                [&gen, &nd] {
                  return nd(gen);
                });
  kernels::Gemm(A.get(), B.get(), 1, N, K, M, C.get());
  kernels::GemmRef(A.get(), B.get(), N, K, M, C_ref.get());

  for (int i = 0; i < N * M; ++i) {
    EXPECT_NEAR(C_ref[i], C[i], 0.1);
  }
}

TEST(GEMMTest, gemv) {
  index_t N = 17;
  index_t K = 63;
  std::unique_ptr<float[]> A(new float[N * K]);
  std::unique_ptr<float[]> B(new float[K]);
  std::unique_ptr<float[]> C(new float[N]);
  std::unique_ptr<float[]> C_ref(new float[N]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  std::generate(A.get(), A.get() + N * K,
                [&gen, &nd] {
                  return nd(gen);
                });
  std::generate(B.get(), B.get() + K,
                [&gen, &nd] {
                  return nd(gen);
                });
  kernels::Gemv(A.get(), B.get(), K, N, C.get());
  kernels::GemvRef(A.get(), B.get(), K, N, C_ref.get());

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(C_ref[i], C[i], 0.1);
  }
}


}  // namespace mace
