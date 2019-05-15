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
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void GatherBenchmark(int iters,
                     index_t n,
                     index_t index_len,
                     index_t vocab_len,
                     index_t embedding_len) {
  mace::testing::StopTiming();
  static unsigned int seed = time(NULL);

  OpsTestNet net;
  std::vector<int32_t> index(index_len);
  for (int i = 0; i < index_len; ++i) {
    index[i] = rand_r(&seed) % vocab_len;
  }
  net.AddInputFromArray<D, int32_t>("Indices", {n, index_len}, index);
  net.AddRandomInput<D, T>("Params", {vocab_len, embedding_len});

  OpDefBuilder("Gather", "GatherTest")
      .Input("Params")
      .Input("Indices")
      .AddIntArg("axis", 0)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_GATHER_MACRO(N, IND, VOC, EMBED, TYPE, DEVICE)            \
  static void                                                             \
      MACE_BM_GATHER##_##N##_##IND##_##VOC##_##EMBED##_##TYPE##_##DEVICE( \
          int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * IND * EMBED;    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    GatherBenchmark<DEVICE, TYPE>(iters, N, IND, VOC, EMBED);             \
  }                                                                       \
  MACE_BENCHMARK(                                                         \
      MACE_BM_GATHER##_##N##_##IND##_##VOC##_##EMBED##_##TYPE##_##DEVICE)

#define MACE_BM_GATHER(N, INDEX, VOCAB, EMBEDDING) \
  MACE_BM_GATHER_MACRO(N, INDEX, VOCAB, EMBEDDING, float, CPU);

MACE_BM_GATHER(1, 7, 48165, 256);
MACE_BM_GATHER(1, 20, 48165, 256);
MACE_BM_GATHER(1, 100, 48165, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
