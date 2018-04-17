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

#include "mace/core/operator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void CWise(int iters, int batch, int channels,
           int height, int width, float x, int type) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("CWise", "CWiseBM")
        .Input("InputImage")
        .Output("Output")
        .AddIntArg("type", type)
        .AddFloatArg("x", x)
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("CWise", "CWiseBM")
        .Input("Input")
        .Output("Output")
        .AddIntArg("type", type)
        .AddFloatArg("x", x)
        .Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}
}  // namespace

#define BM_CWISE_MACRO(N, C, H, W, X, G, TYPE, DEVICE)              \
  static void                                                             \
    BM_CWISE_##N##_##C##_##H##_##W##_##X##_##G##_##TYPE##_##DEVICE( \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::MaccProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    CWise<DEVICE, TYPE>(iters, N, C, H, W, X, G);               \
  }                                                                  \
  BENCHMARK(                                                         \
    BM_CWISE_##N##_##C##_##H##_##W##_##X##_##G##_##TYPE##_##DEVICE)

#define BM_CWISE(N, C, H, W, X, G)                 \
  BM_CWISE_MACRO(N, C, H, W, X, G, float, CPU);    \
  BM_CWISE_MACRO(N, C, H, W, X, G, float, OPENCL); \
  BM_CWISE_MACRO(N, C, H, W, X, G, half, OPENCL);

BM_CWISE(1, 1, 512, 512, 2, 0);
BM_CWISE(1, 3, 128, 128, 2, 1);
BM_CWISE(1, 3, 512, 512, 2, 4);
BM_CWISE(1, 32, 112, 112, 2, 5);
BM_CWISE(1, 32, 112, 112, 2, 6);
BM_CWISE(1, 32, 112, 112, 2, 7);
BM_CWISE(1, 64, 256, 256, 3, 0);
BM_CWISE(1, 64, 512, 512, 3, 1);
BM_CWISE(1, 128, 56, 56, 3, 4);
BM_CWISE(1, 128, 256, 256, 3, 5);
BM_CWISE(1, 64, 512, 512, 3, 6);
BM_CWISE(1, 64, 512, 512, 3, 7);
BM_CWISE(1, 256, 14, 14, 3, 0);
BM_CWISE(1, 512, 14, 14, 3, 1);
BM_CWISE(1, 1024, 7, 7, 3, 4);
BM_CWISE(32, 1, 256, 256, 3, 5);
BM_CWISE(32, 1, 256, 256, 3, 6);
BM_CWISE(32, 1, 256, 256, 3, 7);

}  // namespace test
}  // namespace ops
}  // namespace mace
