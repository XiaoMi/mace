//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_TIMER_H_
#define MACE_UTILS_TIMER_H_

#include "mace/utils/env_time.h"
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

  DISABLE_COPY_AND_ASSIGN(WallClockTimer);
};

}  // namespace mace

#endif  // MACE_UTILS_TIMER_H_
