//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include <thread>

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
  auto TunerFunc = [&](const std::vector<unsigned int> &params) -> int {
    if (params.front() == 1) {
      return expect;
    } else {
      return expect + 1;
    }
  };

  WallClockTimer timer;
  std::vector<unsigned int> default_params(1, 1);
  int res = Tuner<unsigned int>::Get()->template TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);

  EXPECT_EQ(expect, res);

  default_params[0] = 2;
  res = Tuner<unsigned int>::Get()->template TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);
  EXPECT_EQ(expect + 1, res);
}

TEST_F(TunerTest, SimpleTune) {
  int expect = 3;
  auto TunerFunc = [&](const std::vector<unsigned int> &params) -> int {
    if (params.front() == expect) {
      return expect;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return params.front();
    }
  };

  std::vector<unsigned int> default_params(1, 1);
  auto params_generator = []() -> std::vector<std::vector<unsigned int>> {
    return {{1}, {2}, {3}, {4}};
  };
  // tune
  WallClockTimer timer;
  int res = Tuner<unsigned int>::Get()->template TuneOrRun<unsigned int>(
      "SimpleRun", default_params, *params_generator, TunerFunc, &timer);
  EXPECT_EQ(expect, res);

  // run
  res = Tuner<unsigned int>::Get()->template TuneOrRun<unsigned int>(
      "SimpleRun", default_params, nullptr, TunerFunc, &timer);
  EXPECT_EQ(expect, res);
}

}  // namespace mace
