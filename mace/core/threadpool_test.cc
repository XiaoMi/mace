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

#include <gtest/gtest.h>
#include <vector>

#include "mace/core/threadpool.h"
#include "mace/utils/logging.h"

namespace mace {
namespace kernels {
namespace test {

namespace {

void TestParallelRun(int64_t size) {
  ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_NONE);
  std::vector<int64_t> input(size);
  thread_pool.ParallelRun(size, [&](const int64_t start, const int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      input[i] = i;
    }
  });

  for (int64_t i = 0; i < size; ++i) {
    EXPECT_EQ(input[i], i);
  }
}

void TestParallelForLoop1(int64_t start, int64_t end, int64_t step) {
  ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_NONE);
  std::vector<int64_t> input(end);
  thread_pool.ParallelFor(start, end, step, [&](const int64_t i) {
    input[i] = i;
  });

  for (int64_t i = start; i < end; i += step) {
    EXPECT_EQ(input[i], i);
  }
}

void TestParallelForLoop2(int64_t start1, int64_t end1, int64_t step1,
                          int64_t start2, int64_t end2, int64_t step2) {
  ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  std::vector<std::vector<int64_t>> input(end1, std::vector<int64_t>(end2));
  thread_pool.ParallelFor(start1, end1, step1,
                          start2, end2, step2,
                          [&](const int64_t i, const int64_t j) {
                            input[i][j] = i * j;
                          });

  for (int64_t i = start1; i < end1; i += step1) {
    for (int64_t j = start2; j < end2; j += step2) {
      EXPECT_EQ(input[i][j], i * j);
    }
  }
}

void TestParallelForLoop3(int64_t start1, int64_t end1, int64_t step1,
                          int64_t start2, int64_t end2, int64_t step2,
                          int64_t start3, int64_t end3, int64_t step3) {
  ThreadPool thread_pool(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  std::vector<std::vector<std::vector<int64_t>>>
      input(end1,
            std::vector<std::vector<int64_t>>(end2,
                                              std::vector<int64_t>(end3)));
  thread_pool.ParallelFor(start1, end1, step1,
                          start2, end2, step2,
                          start3, end3, step3,
                          [&](const int64_t i,
                              const int64_t j,
                              const int64_t k) {
                            input[i][j][k] = i * j * k;
                          });

  for (int64_t i = start1; i < end1; i += step1) {
    for (int64_t j = start2; j < end2; j += step2) {
      for (int64_t k = start3; k < end3; k += step3) {
        EXPECT_EQ(input[i][j][k], i * j * k);
      }
    }
  }
}

}  // namespace

TEST(ThreadPoolTest, TestParallelRun) {
  TestParallelRun(102);
}

TEST(ThreadPoolTest, TestParallelFor1) {
  TestParallelForLoop1(1, 102, 2);
}

TEST(ThreadPoolTest, TestParallelFor2) {
  TestParallelForLoop2(1, 1, 2, 2, 53, 4);
  TestParallelForLoop2(1, 102, 2, 2, 53, 4);
  TestParallelForLoop2(1, 2, 2, 2, 53, 4);
  TestParallelForLoop2(1, 101, 1, 2, 53, 4);
}

TEST(ThreadPoolTest, TestParallelFor3) {
  TestParallelForLoop3(1, 1, 2, 2, 53, 4, 3, 31, 3);
  TestParallelForLoop3(1, 102, 2, 2, 53, 4, 3, 31, 3);
  TestParallelForLoop3(1, 2, 2, 2, 53, 4, 3, 31, 3);
  TestParallelForLoop3(1, 5, 1, 2, 53, 4, 3, 31, 3);
  TestParallelForLoop3(1, 5, 2, 2, 50, 24, 3, 31, 3);
}

}  // namespace test
}  // namespace kernels
}  // namespace mace
