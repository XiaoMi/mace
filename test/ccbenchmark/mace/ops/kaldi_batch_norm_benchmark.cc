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

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void KaldiBatchNorm(
    int iters, int batch, int chunk, int dim, int block_dim) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, T>("Input", {batch, chunk, dim});
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  net.AddRandomInput<D, T>("Scale", {block_dim}, true);
  net.AddRandomInput<D, T>("Offset", {block_dim}, true);

  OpDefBuilder("KaldiBatchNorm", "KaldiBatchNormBM")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .AddIntArg("block_dim", block_dim)
      .AddIntArg("test_mode", 1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

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

#define MACE_BM_KALDI_BATCH_NORM_MACRO(N, C, D, BD, TYPE, DEVICE)           \
  static void MACE_BM_KALDI_BATCH_NORM_##N##_##C##_##D##_##BD##_##TYPE\
##_##DEVICE(                                                            \
      int iters) {                                                          \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * D;        \
    mace::testing::MacsProcessed(tot);                                      \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                     \
    KaldiBatchNorm<DEVICE, TYPE>(iters, N, C, D, BD);                       \
  }                                                                         \
  MACE_BENCHMARK(MACE_BM_KALDI_BATCH_NORM_##N##_##C##_##D##_##BD##_##TYPE\
##_##DEVICE)

#define MACE_BM_KALDI_BATCH_NORM(N, C, D, BD)                 \
  MACE_BM_KALDI_BATCH_NORM_MACRO(N, C, D, BD, float, CPU);

MACE_BM_KALDI_BATCH_NORM(1, 1, 512, 512);
MACE_BM_KALDI_BATCH_NORM(1, 3, 128, 128);
MACE_BM_KALDI_BATCH_NORM(1, 3, 512, 128);
MACE_BM_KALDI_BATCH_NORM(1, 32, 112, 112);
MACE_BM_KALDI_BATCH_NORM(1, 64, 256, 256);
MACE_BM_KALDI_BATCH_NORM(1, 64, 512, 256);
MACE_BM_KALDI_BATCH_NORM(1, 128, 56, 56);
MACE_BM_KALDI_BATCH_NORM(1, 128, 256, 256);
MACE_BM_KALDI_BATCH_NORM(1, 256, 14, 14);
MACE_BM_KALDI_BATCH_NORM(1, 512, 14, 14);
MACE_BM_KALDI_BATCH_NORM(1, 1024, 7, 7);
MACE_BM_KALDI_BATCH_NORM(32, 1, 256, 128);
MACE_BM_KALDI_BATCH_NORM(32, 3, 256, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
