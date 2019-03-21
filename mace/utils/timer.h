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

#ifndef MACE_UTILS_TIMER_H_
#define MACE_UTILS_TIMER_H_

#include "mace/port/env.h"
#include "mace/utils/logging.h"

namespace mace {

class Timer {
 public:
  virtual void StartTiming() = 0;
  virtual void StopTiming() = 0;
  virtual void AccumulateTiming() = 0;
  virtual void ClearTiming() = 0;
  virtual double ElapsedMicros() = 0;
  virtual double AccumulatedMicros() = 0;
};

class WallClockTimer : public Timer {
 public:
  WallClockTimer() : accumulated_micros_(0) {}

  void StartTiming() override { start_micros_ = NowMicros(); }

  void StopTiming() override { stop_micros_ = NowMicros(); }

  void AccumulateTiming() override {
    StopTiming();
    accumulated_micros_ += stop_micros_ - start_micros_;
  }

  void ClearTiming() override {
    start_micros_ = 0;
    stop_micros_ = 0;
    accumulated_micros_ = 0;
  }

  double ElapsedMicros() override { return stop_micros_ - start_micros_; }

  double AccumulatedMicros() override { return accumulated_micros_; }

 private:
  double start_micros_;
  double stop_micros_;
  double accumulated_micros_;

  MACE_DISABLE_COPY_AND_ASSIGN(WallClockTimer);
};

}  // namespace mace

#endif  // MACE_UTILS_TIMER_H_
