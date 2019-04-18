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

#include <algorithm>
#include <string>
#include <vector>

#include "mace/benchmark_utils/test_benchmark.h"

namespace mace {
namespace ops {
namespace test {

// Test the speed of different access order of a NHWC buffer

namespace {
void MemoryAccessBenchmark_NHWC(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();
  std::vector<float> buffer(batch * height * width * channels);
  std::fill_n(buffer.begin(), buffer.size(), 0.1);
  mace::testing::StartTiming();

  while (iters--) {
    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int c = 0; c < channels; ++c) {
            buffer[n * height * width * channels + h * width * channels +
                   w * channels + c] = 1.0f;
          }
        }
      }
    }
  }
}

void MemoryAccessBenchmark_NWCH(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();
  std::vector<float> buffer(batch * height * width * channels);
  std::fill_n(buffer.begin(), buffer.size(), 0.1);
  mace::testing::StartTiming();

  while (iters--) {
    for (int n = 0; n < batch; ++n) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < height; ++h) {
            buffer[n * height * width * channels + h * width * channels +
                   w * channels + c] = 1.0f;
          }
        }
      }
    }
  }
}

void MemoryAccessBenchmark_NHCW(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();
  std::vector<float> buffer(batch * height * width * channels);
  std::fill_n(buffer.begin(), buffer.size(), 0.1);
  mace::testing::StartTiming();

  while (iters--) {
    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channels; ++c) {
          for (int w = 0; w < width; ++w) {
            buffer[n * height * width * channels + h * width * channels +
                   w * channels + c] = 1.0f;
          }
        }
      }
    }
  }
}

}  // namespace

#define MACE_BM_MEMORY_ACCESS(N, H, W, C, ORDER)                     \
  static void MACE_BM_MEMORY_ACCESS_##N##_##H##_##W##_##C##_##ORDER( \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C; \
    mace::testing::BytesProcessed(tot * sizeof(float));              \
    MemoryAccessBenchmark_##ORDER(iters, N, H, W, C);                \
  }                                                                  \
  MACE_BENCHMARK(MACE_BM_MEMORY_ACCESS_##N##_##H##_##W##_##C##_##ORDER)

MACE_BM_MEMORY_ACCESS(10, 64, 64, 1024, NHWC);
MACE_BM_MEMORY_ACCESS(10, 64, 64, 1024, NHCW);
MACE_BM_MEMORY_ACCESS(10, 64, 64, 1024, NWCH);
MACE_BM_MEMORY_ACCESS(10, 64, 1024, 64, NHCW);
MACE_BM_MEMORY_ACCESS(10, 64, 1024, 64, NWCH);

}  // namespace test
}  // namespace ops
}  // namespace mace
