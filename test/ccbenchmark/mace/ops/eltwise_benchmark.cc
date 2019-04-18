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

#include <string>

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/eltwise.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void EltwiseBenchmark(
    int iters, ops::EltwiseType type, int n, int h, int w, int c) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  if (D == DeviceType::CPU && DataTypeToEnum<T>::value != DT_UINT8) {
    net.AddRandomInput<D, T>("Input0", {n, c, h, w});
    net.AddRandomInput<D, T>("Input1", {n, c, h, w});
  } else {
    net.AddRandomInput<D, T>("Input0", {n, h, w, c});
    net.AddRandomInput<D, T>("Input1", {n, h, w, c});
  }

  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", {1.2, 2.1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("has_data_format", 1)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.Setup(D);

  if (D == DeviceType::CPU && DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, TYPE, DEVICE)             \
  static void                                                                 \
      MACE_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;          \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    EltwiseBenchmark<DEVICE, TYPE>(                                           \
        iters, static_cast<ops::EltwiseType>(ELT_TYPE), N, H, W, C);      \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_ELTWISE(ELT_TYPE, N, H, W, C)                 \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, CPU);    \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, GPU);    \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, half, GPU)
#else
#define MACE_BM_ELTWISE(ELT_TYPE, N, H, W, C)                 \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, CPU)
#endif

MACE_BM_ELTWISE(2, 1, 128, 128, 32);
MACE_BM_ELTWISE(2, 1, 240, 240, 256);
MACE_BM_ELTWISE(2, 1, 256, 256, 32);
MACE_BM_ELTWISE(0, 1, 128, 128, 32);
MACE_BM_ELTWISE(0, 1, 240, 240, 256);
MACE_BM_ELTWISE(5, 1, 128, 128, 32);
MACE_BM_ELTWISE(5, 1, 240, 240, 256);

#ifdef MACE_ENABLE_QUANTIZE
MACE_BM_ELTWISE_MACRO(0, 1, 128, 128, 32, uint8_t, CPU);
MACE_BM_ELTWISE_MACRO(1, 1, 128, 128, 32, uint8_t, CPU);
#endif

}  // namespace test
}  // namespace ops
}  // namespace mace
