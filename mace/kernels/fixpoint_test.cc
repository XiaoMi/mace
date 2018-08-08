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
#include <random>
#include <vector>
#include <algorithm>

#include "mace/kernels/fixpoint.h"

namespace mace {
namespace kernels {
namespace test {

namespace {
void TestFindMax(int test_count) {
  static unsigned int seed = time(NULL);
  std::vector<uint8_t> input(test_count);
  uint8_t expected_max = 0;
  for (int i = 0; i < test_count; ++i) {
    input[i] = rand_r(&seed) % 255;
    expected_max = std::max(expected_max, input[i]);
  }

  uint8_t actual_max = FindMax(input.data(), input.size());
  EXPECT_EQ(expected_max, actual_max);
}
}  // namespace

TEST(FixpointTest, FindMax) {
  TestFindMax(1);
  TestFindMax(2);
  TestFindMax(4);
  TestFindMax(8);
  TestFindMax(32);
  TestFindMax(33);
  TestFindMax(127);
}

}  // namespace test
}  // namespace kernels
}  // namespace mace

