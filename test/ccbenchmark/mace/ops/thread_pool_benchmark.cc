// Copyright 2019 The MACE Authors. All Rights Reserved.
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

// OpenMP and Mace thread pool should be benchmarked separately.

#include <algorithm>
#include <string>
#include <vector>

#include "mace/core/types.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/utils/thread_pool.h"

#define MACE_EMPTY_STATEMENT asm volatile("":::"memory");

namespace mace {
namespace ops {
namespace test {

namespace {

const index_t kMaxSize = 100000000;
const index_t image_size = 56 * 56;
std::vector<float> output_data(kMaxSize), bias_data(kMaxSize);

void OpenMPBenchmark1D(int iters, int size) {
  while (iters--) {
    const int b = 0;
#pragma omp parallel for schedule(runtime)
    for (int c = 0; c < size; ++c) {
      for (index_t i = 0; i < image_size; ++i) {
        output_data[(b * size + c) * image_size + i] += bias_data[c];
      }
    }
  }
}

void ThreadPoolBenchmark1D(int iters, int size) {
  mace::testing::StopTiming();
  utils::ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  thread_pool.Init();
  mace::testing::StartTiming();

  while (iters--) {
    const int b = 0;
    thread_pool.Compute1D([=](index_t start0, index_t end0, index_t step0) {
      for (index_t c = start0; c < end0; c += step0) {
        for (index_t i = 0; i < image_size; ++i) {
          output_data[(b * size + c) * image_size + i] += bias_data[c];
        }
      }
    }, 0, size, 1);
  }
}

void OpenMPBenchmark2D(int iters, int size0, int size1) {
  while (iters--) {
#pragma omp parallel for collapse(2) schedule(runtime)
    for (int b = 0; b < size0; ++b) {
      for (int c = 0; c < size1; ++c) {
        for (index_t i = 0; i < image_size; ++i) {
          output_data[(b * size1 + c) * image_size + i] += bias_data[c];
        }
      }
    }
  }
}

void ThreadPoolBenchmark2D(int iters, int size0, int size1) {
  mace::testing::StopTiming();
  utils::ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  thread_pool.Init();
  mace::testing::StartTiming();

  while (iters--) {
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t c = start1; c < end1; c += step1) {
          for (index_t i = 0; i < image_size; ++i) {
            output_data[(b * size1 + c) * image_size + i] += bias_data[c];
          }
        }
      }
    }, 0, size0, 1, 0, size1, 1);
  }
}

}  // namespace

#define MACE_BM_THREADPOOL_OPENMP_1D(SIZE)                               \
  static void MACE_BM_THREADPOOL_OPENMP_1D_##SIZE(int iters) {           \
    const int64_t tot = static_cast<int64_t>(iters) * SIZE;              \
    mace::testing::MacsProcessed(static_cast<int64_t>(iters) * SIZE);    \
    mace::testing::BytesProcessed(tot * sizeof(float));                  \
    OpenMPBenchmark1D(iters, SIZE);                                      \
  }                                                                      \
  MACE_BENCHMARK(MACE_BM_THREADPOOL_OPENMP_1D_##SIZE)

#define MACE_BM_THREADPOOL_MACE_1D(SIZE)                                 \
  static void MACE_BM_THREADPOOL_MACE_1D_##SIZE(int iters) {             \
    const int64_t tot = static_cast<int64_t>(iters) * SIZE;              \
    mace::testing::MacsProcessed(static_cast<int64_t>(iters) * SIZE);    \
    mace::testing::BytesProcessed(tot * sizeof(float));                  \
    ThreadPoolBenchmark1D(iters, SIZE);                                  \
  }                                                                      \
  MACE_BENCHMARK(MACE_BM_THREADPOOL_MACE_1D_##SIZE)

#define MACE_BM_THREADPOOL_OPENMP_2D(SIZE0, SIZE1)                            \
  static void MACE_BM_THREADPOOL_OPENMP_2D_##SIZE0##_##SIZE1(int iters) {     \
    const int64_t tot = static_cast<int64_t>(iters) * SIZE0 * SIZE1;          \
    mace::testing::MacsProcessed(static_cast<int64_t>(iters) * SIZE0 * SIZE1);\
    mace::testing::BytesProcessed(tot * sizeof(float));                       \
    OpenMPBenchmark2D(iters, SIZE0, SIZE1);                                   \
  }                                                                           \
  MACE_BENCHMARK(MACE_BM_THREADPOOL_OPENMP_2D_##SIZE0##_##SIZE1)

#define MACE_BM_THREADPOOL_MACE_2D(SIZE0, SIZE1)                              \
  static void MACE_BM_THREADPOOL_MACE_2D_##SIZE0##_##SIZE1(int iters) {       \
    const int64_t tot = static_cast<int64_t>(iters) * SIZE0 * SIZE1;          \
    mace::testing::MacsProcessed(static_cast<int64_t>(iters) * SIZE0 * SIZE1);\
    mace::testing::BytesProcessed(tot * sizeof(float));                       \
    ThreadPoolBenchmark2D(iters, SIZE0, SIZE1);                               \
  }                                                                           \
  MACE_BENCHMARK(MACE_BM_THREADPOOL_MACE_2D_##SIZE0##_##SIZE1)

// OpenMP and Mace threadpool need to be benchmarked separately.

MACE_BM_THREADPOOL_OPENMP_1D(64);
MACE_BM_THREADPOOL_OPENMP_1D(128);
MACE_BM_THREADPOOL_OPENMP_1D(256);
MACE_BM_THREADPOOL_OPENMP_1D(512);
MACE_BM_THREADPOOL_OPENMP_1D(1024);

MACE_BM_THREADPOOL_OPENMP_2D(1, 64);
MACE_BM_THREADPOOL_OPENMP_2D(1, 128);
MACE_BM_THREADPOOL_OPENMP_2D(1, 256);
MACE_BM_THREADPOOL_OPENMP_2D(1, 512);
MACE_BM_THREADPOOL_OPENMP_2D(1, 1024);


MACE_BM_THREADPOOL_MACE_1D(64);
MACE_BM_THREADPOOL_MACE_1D(128);
MACE_BM_THREADPOOL_MACE_1D(256);
MACE_BM_THREADPOOL_MACE_1D(512);
MACE_BM_THREADPOOL_MACE_1D(1024);

MACE_BM_THREADPOOL_MACE_2D(1, 64);
MACE_BM_THREADPOOL_MACE_2D(1, 128);
MACE_BM_THREADPOOL_MACE_2D(1, 256);
MACE_BM_THREADPOOL_MACE_2D(1, 512);
MACE_BM_THREADPOOL_MACE_2D(1, 1024);

}  // namespace test
}  // namespace ops
}  // namespace mace
