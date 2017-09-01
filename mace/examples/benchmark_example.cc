//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"

static void foo(int iters) {
  static const int N = 32;
  const int64 tot = static_cast<int64>(iters) * N;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  float* inp = new float[N];
  float* out = new float[N];

  while (iters--) {
    for (int i=0; i < N; i++) {
      out[i] = inp[i] * 2.0;
    }
  }
  delete[] inp;
  delete[] out;
}

BENCHMARK(foo);


static void bar(int iters, int n) {
  const int64 tot = static_cast<int64>(iters) * n;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  float* inp = new float[n];
  float* out = new float[n];

  while (iters--) {
    for (int i=0; i < n; i++) {
      out[i] = inp[i] * 2.0;
    }
  }
  delete[] inp;
  delete[] out;
}

BENCHMARK(bar)->Arg(32)->Arg(64)->Arg(128);
