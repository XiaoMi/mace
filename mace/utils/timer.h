//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_TIMER_H_
#define MACE_UTILS_TIMER_H_

#include "mace/utils/env_time.h"

namespace mace {

class Timer {
  public:
    virtual void StartTiming() = 0;
    virtual void StopTiming() = 0;
    virtual double ElapsedMicros() = 0;
};

class WallClockTimer : public Timer {
  public:
    void StartTiming() override {
      start_micros_ = mace::utils::NowMicros();
    }

    void StopTiming() override {
      stop_micros_ = mace::utils::NowMicros();
    }

    double ElapsedMicros() override {
      return stop_micros_ - start_micros_;
    }

  private:
    double start_micros_;
    double stop_micros_;
};

}  // namespace mace

#endif  // MACE_UTILS_TIMER_H_
