// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifdef MACE_ENABLE_QUANTIZE

#include "mace/core/operator.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void Quantize(int iters, int count) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", {count});

  OpDefBuilder("Quantize", "QuantizeBM")
      .Input("Input")
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}

template <DeviceType D, typename T>
void Dequantize(int iters, int count) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, T>("Input", {count});

  OpDefBuilder("Dequantize", "DequantizeBM")
      .Input("Input")
      .Output("Output")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 2; ++i) {
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

#define MACE_BM_QUANTIZE_MACRO(N, TYPE, DEVICE)            \
  static void                                              \
    MACE_BM_QUANTIZE_##N##_##TYPE##_##DEVICE(              \
      int iters) {                                         \
    const int64_t tot = static_cast<int64_t>(iters) * N;   \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));    \
    Quantize<DEVICE, TYPE>(iters, N);                      \
  }                                                        \
  MACE_BENCHMARK(                                          \
    MACE_BM_QUANTIZE_##N##_##TYPE##_##DEVICE)

#define MACE_BM_QUANTIZE(N)                                \
  MACE_BM_QUANTIZE_MACRO(N, uint8_t, CPU);

#define MACE_BM_DEQUANTIZE_MACRO(N, TYPE, DEVICE)          \
  static void                                              \
    MACE_BM_DEQUANTIZE_##N##_##TYPE##_##DEVICE(            \
      int iters) {                                         \
    const int64_t tot = static_cast<int64_t>(iters) * N;   \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));    \
    Dequantize<DEVICE, TYPE>(iters, N);                    \
  }                                                        \
  MACE_BENCHMARK(                                          \
    MACE_BM_DEQUANTIZE_##N##_##TYPE##_##DEVICE)

#define MACE_BM_DEQUANTIZE(N)                              \
  MACE_BM_DEQUANTIZE_MACRO(N, uint8_t, CPU);

MACE_BM_QUANTIZE(256);
MACE_BM_QUANTIZE(1470000);
MACE_BM_DEQUANTIZE(256);
MACE_BM_DEQUANTIZE(1470000);

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_ENABLE_QUANTIZE
