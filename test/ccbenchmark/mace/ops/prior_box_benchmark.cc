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
template<DeviceType D, typename T>
void PriorBox(
    int iters, float min_size, float max_size, float aspect_ratio,
    int clip, float variance0, float variance1, float offset, int h) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("INPUT", {1, h, 1, 1});
  net.AddRandomInput<D, float>("DATA", {1, 3, 300, 300});

  OpDefBuilder("PriorBox", "PriorBoxBM")
      .Input("INPUT")
      .Input("DATA")
      .Output("OUTPUT")
      .AddFloatsArg("min_size", {min_size})
      .AddFloatsArg("max_size", {max_size})
      .AddFloatsArg("aspect_ratio", {aspect_ratio})
      .AddIntArg("clip", clip)
      .AddFloatsArg("variance", {variance0, variance0, variance1, variance1})
      .AddFloatArg("offset", offset)
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  const int64_t tot = static_cast<int64_t>(iters) * (300 * 300 * 3 + h);
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}
}  // namespace

#define MACE_BM_PRIOR_BOX(MIN, MAX, AR, CLIP, V0, V1, OFFSET, H)              \
  static void MACE_BM_PRIOR_BOX_##MIN##_##MAX##_##AR##_##CLIP##_##V0##_##V1##_\
      ##OFFSET##_##H(int iters) {                                             \
    PriorBox<DeviceType::CPU, float>(iters, MIN, MAX, AR, CLIP, V0, V1,       \
        OFFSET, H);                                                           \
  }                                                                           \
  MACE_BENCHMARK(MACE_BM_PRIOR_BOX_##MIN##_##MAX##_##AR##_##CLIP##_##V0##_    \
      ##V1##_##OFFSET##_##H)

MACE_BM_PRIOR_BOX(285, 300, 2, 0, 1, 2, 1, 128);

}  // namespace test
}  // namespace ops
}  // namespace mace

