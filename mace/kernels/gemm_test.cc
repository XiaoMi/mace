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

#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "mace/core/types.h"
#include "mace/kernels/gemm.h"

namespace mace {

namespace {

void GemmTest(index_t batch,
              index_t N,
              index_t K,
              index_t M,
              bool transpose_a,
              bool transpose_b) {
  std::unique_ptr<float[]> A(new float[batch * N * K]);
  std::unique_ptr<float[]> B(new float[batch * K * M]);
  std::unique_ptr<float[]> C(new float[batch * N * M]);
  std::unique_ptr<float[]> C_ref(new float[batch * N * M]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  std::generate(A.get(), A.get() + batch * N * K,
                [&gen, &nd] { return nd(gen); });
  std::generate(B.get(), B.get() + batch * K * M,
                [&gen, &nd] { return nd(gen); });
  kernels::Gemm(A.get(), B.get(), batch, N, K, M, C.get(), transpose_a,
                transpose_b);
  kernels::GemmRef(A.get(), B.get(), batch, N, K, M, C_ref.get(), transpose_a,
                   transpose_b);

  for (int i = 0; i < batch * N * M; ++i) {
    EXPECT_NEAR(C_ref[i], C[i], 0.1);
  }
}

void GemvTest(index_t batch, index_t N, index_t M) {
  std::unique_ptr<float[]> A(new float[N * M]);
  std::unique_ptr<float[]> B(new float[batch * M]);
  std::unique_ptr<float[]> C(new float[batch * N]);
  std::unique_ptr<float[]> C_ref(new float[batch * N]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  std::generate(A.get(), A.get() + N * M, [&gen, &nd] { return nd(gen); });
  std::generate(B.get(), B.get() + batch * M, [&gen, &nd] { return nd(gen); });
  kernels::Gemv(A.get(), B.get(), batch, M, N, C.get());
  kernels::GemvRef(A.get(), B.get(), batch, M, N, C_ref.get());

  for (int i = 0; i < batch * N; ++i) {
    EXPECT_NEAR(C_ref[i], C[i], 0.1);
  }
}

}  // namespace

TEST(GEMMTest, AlignedWithoutBatch) {
  GemmTest(1, 1, 64, 128, false, false);
  GemmTest(1, 2, 64, 128, false, true);
  GemmTest(1, 3, 64, 128, true, false);
  GemmTest(1, 4, 64, 128, true, true);
  GemmTest(1, 5, 64, 128, false, false);
  GemmTest(1, 6, 64, 128, false, true);
  GemmTest(1, 7, 64, 128, true, false);
  GemmTest(1, 17, 64, 128, true, true);
}

TEST(GEMMTest, UnalignedWithoutBatch) {
  GemmTest(1, 1, 63, 127, false, false);
  GemmTest(1, 2, 63, 127, false, true);
  GemmTest(1, 3, 63, 127, true, false);
  GemmTest(1, 4, 63, 127, true, true);
  GemmTest(1, 5, 63, 127, false, false);
  GemmTest(1, 6, 63, 127, false, true);
  GemmTest(1, 7, 63, 127, true, false);
  GemmTest(1, 17, 63, 127, true, true);
}

TEST(GEMMTest, UnalignedWithBatch) {
  GemmTest(3, 1, 63, 127, false, false);
  GemmTest(3, 2, 63, 127, false, true);
  GemmTest(3, 3, 63, 127, true, false);
  GemmTest(3, 4, 63, 127, true, true);
  GemmTest(3, 5, 63, 127, false, false);
  GemmTest(3, 6, 63, 127, false, true);
  GemmTest(3, 7, 63, 127, true, false);
  GemmTest(3, 17, 63, 127, true, true);
}

TEST(GEMMTest, gemv) {
  GemvTest(1, 17, 63);
  GemvTest(3, 17, 63);
}

}  // namespace mace
