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

#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>
#include "mace/utils/thread_pool.h"

namespace mace {
namespace utils {
namespace {

class ThreadPoolTest : public ::testing::Test {
 public:
  ThreadPoolTest()
      : thread_pool(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
    thread_pool.Init();
  }
  ThreadPool thread_pool;
};

void Test1D(int64_t start, int64_t end, int64_t step, std::vector<int> *res) {
  for (int64_t i = start; i < end; i += step) {
    (*res)[i]++;
  }
}

void Test2D(int64_t start0, int64_t end0, int64_t step0,
            int64_t start1, int64_t end1, int64_t step1,
            std::vector<int> *res) {
  for (int64_t i = start0; i < end0; i += step0) {
    for (int64_t j = start1; j < end1; j += step1) {
      (*res)[i * 100 + j]++;
    }
  }
}

void Test3D(int64_t start0, int64_t end0, int64_t step0,
            int64_t start1, int64_t end1, int64_t step1,
            int64_t start2, int64_t end2, int64_t step2,
            std::vector<int> *res) {
  for (int64_t i = start0; i < end0; i += step0) {
    for (int64_t j = start1; j < end1; j += step1) {
      for (int64_t k = start2; k < end2; k += step2) {
        (*res)[(i * 100 + j) * 100 + k]++;
      }
    }
  }
}

TEST_F(ThreadPoolTest, Compute1D) {
  int64_t test_size = 100;
  std::vector<int> actual(test_size, 0);
  thread_pool.Compute1D([&](int64_t start, int64_t end, int64_t step) {
    Test1D(start, end, step, &actual);
  }, 0, test_size, 2);
  std::vector<int> expected(test_size, 0);
  Test1D(0, test_size, 2, &expected);

  for (int64_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

TEST_F(ThreadPoolTest, Compute2D) {
  int64_t test_size = 100;
  std::vector<int> actual(test_size * test_size, 0);
  thread_pool.Compute2D([&](int64_t start0, int64_t end0, int64_t step0,
                             int64_t start1, int64_t end1, int64_t step1) {
    Test2D(start0, end0, step0, start1, end1, step1, &actual);
  }, 0, test_size, 2, 0, test_size, 2);
  std::vector<int> expected(test_size * test_size, 0);
  Test2D(0, test_size, 2, 0, test_size, 2, &expected);

  for (int64_t i = 0; i < test_size * test_size; ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

TEST_F(ThreadPoolTest, Compute3D) {
  int64_t test_size = 100;
  std::vector<int> actual(test_size * test_size * test_size, 0);
  thread_pool.Compute3D([&](int64_t start0, int64_t end0, int64_t step0,
                             int64_t start1, int64_t end1, int64_t step1,
                             int64_t start2, int64_t end2, int64_t step2) {
    Test3D(start0, end0, step0, start1, end1, step1, start2, end2, step2,
           &actual);
  }, 0, test_size, 2, 0, test_size, 2, 0, test_size, 2);
  std::vector<int> expected(test_size * test_size * test_size, 0);
  Test3D(0, test_size, 2, 0, test_size, 2, 0, test_size, 2, &expected);

  for (int64_t i = 0; i < test_size * test_size * test_size; ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

}  // namespace
}  // namespace utils
}  // namespace mace
