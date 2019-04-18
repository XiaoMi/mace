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

#include <thread>  // NOLINT(build/c++11)

#include "gtest/gtest.h"

#include "mace/utils/tuner.h"

namespace mace {

class TunerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    remove("/data/local/tmp/mace.config");
    setenv("MACE_RUN_PARAMETER_PATH", "/data/local/tmp/mace.config", 1);
    setenv("MACE_TUNING", "1", 1);
  }
};

TEST_F(TunerTest, SimpleRun) {
  int expect = 1;
  auto TunerFunc = [&](const std::vector<unsigned int> &params, Timer *timer,
                       std::vector<uint32_t> *tuning_result) -> int {
    (void)(timer);
    (void)(tuning_result);
    if (params.front() == 1) {
      return expect;
    } else {
      return expect + 1;
    }
  };

  Tuner<unsigned int> tuner;
  WallClockTimer timer;
  std::vector<unsigned int> default_params(1, 1);
  int res = tuner.TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);

  EXPECT_EQ(expect, res);

  default_params[0] = 2;
  res = tuner.TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);
  EXPECT_EQ(expect + 1, res);
}

TEST_F(TunerTest, SimpleTune) {
  unsigned int expect = 3;
  auto TunerFunc = [&](const std::vector<unsigned int> &params, Timer *timer,
                       std::vector<uint32_t> *tuning_result) -> int {
    int res = 0;
    if (timer) {
      timer->ClearTiming();
      timer->StartTiming();
      if (params.front() == expect) {
        timer->AccumulateTiming();
        res = expect;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timer->AccumulateTiming();
        res = params.front();
      }
      tuning_result->assign(params.begin(), params.end());
    } else {
      if (params.front() == expect) {
        res = expect;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        res = params.front();
      }
    }
    return res;
  };

  std::vector<unsigned int> default_params(1, 1);
  auto params_generator = []() -> std::vector<std::vector<unsigned int>> {
    return {{1}, {2}, {3}, {4}};
  };
  // tune
  Tuner<unsigned int> tuner;
  WallClockTimer timer;
  int res = tuner.TuneOrRun<unsigned int>(
      "SimpleRun", default_params, *params_generator, TunerFunc, &timer);
  EXPECT_EQ(expect, res);

  // run
  res = tuner.template TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);
  EXPECT_EQ(expect, res);
}

}  // namespace mace
