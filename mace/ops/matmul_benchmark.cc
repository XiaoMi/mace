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

#include <string>

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void MatMulBenchmark(
    int iters, int batch, int height, int channels, int out_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("A", {batch, height, channels});
  net.AddRandomInput<D, T>("B", {batch, channels, out_width});
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("A")->SetScale(0.1);
    net.GetTensor("B")->SetScale(0.1);
  }

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "A", "AImage", kernels::BufferType::IN_OUT_WIDTH);
    BufferToImage<D, T>(&net, "B", "BImage",
                        kernels::BufferType::IN_OUT_HEIGHT);

    OpDefBuilder("MatMul", "MatMulBM")
        .Input("AImage")
        .Input("BImage")
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("MatMul", "MatMulBM")
        .Input("A")
        .Input("B")
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  }

  net.Setup(D);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}

template <DeviceType D, typename T>
void MatMulTransposeBenchmark(
    int iters, int batch, int height, int channels, int out_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("A", {batch, height, channels});
  net.AddRandomInput<D, T>("B", {batch, out_width, channels});
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("A")->SetScale(0.1);
    net.GetTensor("B")->SetScale(0.1);
  }

  if (D == DeviceType::CPU) {
    OpDefBuilder("MatMul", "MatMulBM")
        .Input("A")
        .Input("B")
        .AddIntArg("transpose_b", 1)
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  net.Setup(D);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_MATMUL_MACRO(N, H, C, W, TYPE, DEVICE)                         \
  static void MACE_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                             \
    const int64_t macc = static_cast<int64_t>(iters) * N * C * H * W;          \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W);     \
    mace::testing::MaccProcessed(macc);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    MatMulBenchmark<DEVICE, TYPE>(iters, N, H, C, W);                          \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_MATMUL(N, H, C, W)                 \
  MACE_BM_MATMUL_MACRO(N, H, C, W, float, CPU);    \
  MACE_BM_MATMUL_MACRO(N, H, C, W, float, GPU);    \
  MACE_BM_MATMUL_MACRO(N, H, C, W, half, GPU);     \
  MACE_BM_MATMUL_MACRO(N, H, C, W, uint8_t, CPU);

#define MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, TYPE, DEVICE)               \
  static void MACE_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE(    \
      int iters) {                                                             \
    const int64_t macc = static_cast<int64_t>(iters) * N * C * H * W;          \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W);     \
    mace::testing::MaccProcessed(macc);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    MatMulTransposeBenchmark<DEVICE, TYPE>(iters, N, H, C, W);                 \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_MATMUL_TRANPOSE(N, H, C, W)                   \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, float, CPU);     \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, uint8_t, CPU);

MACE_BM_MATMUL(16, 32, 128, 49);
MACE_BM_MATMUL(16, 32, 128, 961);
MACE_BM_MATMUL(16, 32, 128, 3969);
MACE_BM_MATMUL(16, 128, 128, 49);
MACE_BM_MATMUL(16, 128, 128, 961);
MACE_BM_MATMUL(16, 128, 128, 3969);

MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 49);
MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 961);
MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 3969);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 49);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 961);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 3969);

}  // namespace test
}  // namespace ops
}  // namespace mace
