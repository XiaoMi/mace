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

#include "mace/port/logger.h"

#include <gtest/gtest.h>

namespace mace {
namespace {

class LoggerTest : public ::testing::Test {
};

TEST_F(LoggerTest, LogLevel) {
  EXPECT_EQ(INFO, port::LogLevelFromStr("i"));
  EXPECT_EQ(INFO, port::LogLevelFromStr("I"));
  EXPECT_EQ(INFO, port::LogLevelFromStr("INFO"));

  EXPECT_EQ(WARNING, port::LogLevelFromStr("w"));
  EXPECT_EQ(WARNING, port::LogLevelFromStr("W"));
  EXPECT_EQ(WARNING, port::LogLevelFromStr("WARNING"));

  EXPECT_EQ(ERROR, port::LogLevelFromStr("e"));
  EXPECT_EQ(ERROR, port::LogLevelFromStr("E"));
  EXPECT_EQ(ERROR, port::LogLevelFromStr("ERROR"));

  EXPECT_EQ(FATAL, port::LogLevelFromStr("f"));
  EXPECT_EQ(FATAL, port::LogLevelFromStr("F"));
  EXPECT_EQ(FATAL, port::LogLevelFromStr("FATAL"));
}

}  // namespace
}  // namespace mace
