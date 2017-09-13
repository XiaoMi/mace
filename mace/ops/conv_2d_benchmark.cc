//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/conv_2d.h"

namespace mace {

template <DeviceType D, typename T>
static void Conv2d(int iters, int batch, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride,
                   Padding padding, int output_channels) {
  mace::testing::StopTiming();

  mace::testing::StartTiming();
  while(iters--) {
  }
}

#define BM_CONV_2D_MACRO(N, C, H, W, KH, KW, STRIDE, P, OC, TYPE, DEVICE) \
  static void BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_OC##_##TYPE##_##DEVICE(  \
        int iters) {                                                               \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;               \
    mace::testing::ItemsProcessed(tot);                                            \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));                           \
    Conv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, mace::Padding::P, OC); \
  }                                                                                \
  BENCHMARK(BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_OC##_##TYPE##_##DEVICE)

#define BM_CONV_2D(N, C, H, W, KH, KW, S, P, OC, TYPE)        \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, TYPE, CPU);  \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, TYPE, NEON);

BM_CONV_2D(1, 64, 32, 32, 1, 1, 1, VALID, 128, float);

} //  namespace mace
