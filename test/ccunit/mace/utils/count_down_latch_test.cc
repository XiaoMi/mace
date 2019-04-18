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
#include <numeric>
#include <limits>
#include <vector>

#include "mace/utils/count_down_latch.h"

namespace mace {
namespace utils {

namespace {

class CountDownLatchTest : public ::testing::Test {
};

TEST_F(CountDownLatchTest, TestWait) {
  CountDownLatch latch(0, 10);
  std::vector<std::thread> threads(10);
  for (int i = 0; i < 10; ++i) {
    threads[i] = std::thread([&latch]() {
      latch.CountDown();
    });
  }

  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }
  MACE_CHECK(latch.count() == 0);
}

TEST_F(CountDownLatchTest, TestSpinWait) {
  CountDownLatch latch(100, 10);
  std::vector<std::thread> threads(10);
  for (int i = 0; i < 10; ++i) {
    threads[i] = std::thread([&latch]() {
      latch.CountDown();
    });
  }

  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }
  MACE_CHECK(latch.count() == 0);
}

}  // namespace

}  // namespace utils
}  // namespace mace
