//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/batch_norm.h"

namespace mace {
template <DeviceType D, typename T>
static void BatchNorm(int iters, int batch, int channels, int height, int width) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  TIndex input_size = batch * channels * height * width;
  std::vector<T> input(input_size, 0.0);
  std::vector<T> scale(channels, 0.0);
  std::vector<T> offset(channels, 0.0);
  std::vector<T> mean(channels, 0.0);
  std::vector<T> var(channels, 0.0);

  for (int i = 0; i < input_size; ++i) {
    input[i] = nd(gen);
  }
  for (int i = 0; i < channels; ++i) {
    scale[i] = nd(gen);
    offset[i] = nd(gen);
    mean[i] = nd(gen);
    var[i] = std::abs(nd(gen));
  }

  // declare output
  std::unique_ptr<T[]> output(new T[input_size]);
  auto functor = kernels::BatchNormFunctor<D, T>(1e-5);

  while(iters--) {
    functor(input.data(),
        scale.data(),
        offset.data(),
        mean.data(),
        var.data(),
        batch,
        channels,
        height * width,
        output.get());
  }
}

#define BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, DEVICE)                  \
  static void BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
        int iters) {                                                    \
    const int64 tot = static_cast<int64>(iters) * N * C * H * W;                        \
    mace::testing::ItemsProcessed(tot);                                 \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));\
    BatchNorm<DEVICE, TYPE>(iters, N, C, H, W);                         \
  }                                                                     \
  BENCHMARK(BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_BATCH_NORM(N, C, H, W, TYPE)                    \
  BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, CPU);  \
  BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, NEON);

BM_BATCH_NORM(1, 1, 128, 128, float);
BM_BATCH_NORM(1, 1, 512, 512, float);
BM_BATCH_NORM(1, 1, 1024, 1024, float);
BM_BATCH_NORM(16, 1, 256, 256, float);
BM_BATCH_NORM(32, 1, 256, 256, float);
BM_BATCH_NORM(64, 1, 256, 256, float);
BM_BATCH_NORM(1, 3, 128, 128, float);
BM_BATCH_NORM(1, 3, 512, 512, float);
BM_BATCH_NORM(1, 3, 1024, 1024, float);
BM_BATCH_NORM(16, 3, 256, 256, float);
BM_BATCH_NORM(32, 3, 256, 256, float);
BM_BATCH_NORM(64, 3, 256, 256, float);
} //  namespace mace