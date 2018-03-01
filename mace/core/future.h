//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_FUTURE_H_
#define MACE_CORE_FUTURE_H_

#include <functional>

#include "mace/utils/logging.h"

namespace mace {

class CallStats;

// Wait the call to finish and get the stats if param is not nullptr
struct StatsFuture {
  std::function<void(CallStats *)> wait_fn = [](CallStats *) {
    LOG(FATAL) << "wait_fn must be properly set";
  };
};

}  // namespace mace

#endif  // MACE_CORE_FUTURE_H_
