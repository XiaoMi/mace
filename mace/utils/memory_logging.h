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

#ifndef MACE_UTILS_MEMORY_LOGGING_H_
#define MACE_UTILS_MEMORY_LOGGING_H_

#ifndef __hexagon__
#include <malloc.h>
#endif
#include <string>

#include "mace/utils/logging.h"

namespace mace {

#ifdef MACE_ENABLE_MEMORY_LOGGING
class MallinfoChangeLogger {
 public:
  explicit MallinfoChangeLogger(const std::string &name) : name_(name) {
    prev_ = mallinfo();
  }
  ~MallinfoChangeLogger() {
    struct mallinfo curr = mallinfo();
    LogMallinfoChange(name_, curr, prev_);
  }

 private:
  const std::string name_;
  struct mallinfo prev_;

  struct mallinfo LogMallinfoChange(const std::string &name,
                                    const struct mallinfo curr,
                                    const struct mallinfo prev) {
    if (prev.arena != curr.arena) {
      LOG(INFO) << "[" << name << "] "
                << "Non-mmapped space allocated (bytes): " << curr.arena
                << ", diff: " << ((int64_t)curr.arena - (int64_t)prev.arena);
    }
    if (prev.ordblks != curr.ordblks) {
      LOG(INFO) << "[" << name << "] "
                << "Number of free chunks: " << curr.ordblks << ", diff: "
                << ((int64_t)curr.ordblks - (int64_t)prev.ordblks);
    }
    if (prev.smblks != curr.smblks) {
      LOG(INFO) << "[" << name << "] "
                << "Number of free fastbin blocks: " << curr.smblks
                << ", diff: " << ((int64_t)curr.smblks - (int64_t)prev.smblks);
    }
    if (prev.hblks != curr.hblks) {
      LOG(INFO) << "[" << name << "] "
                << "Number of mmapped regions: " << curr.hblks
                << ", diff: " << ((int64_t)curr.hblks - (int64_t)prev.hblks);
    }
    if (prev.hblkhd != curr.hblkhd) {
      LOG(INFO) << "[" << name << "] "
                << "Space allocated in mmapped regions (bytes): " << curr.hblkhd
                << ", diff: " << ((int64_t)curr.hblkhd - (int64_t)prev.hblkhd);
    }
    if (prev.usmblks != curr.usmblks) {
      LOG(INFO) << "[" << name << "] "
                << "Maximum total allocated space (bytes): " << curr.usmblks
                << ", diff: "
                << ((int64_t)curr.usmblks - (int64_t)prev.usmblks);
    }
    if (prev.fsmblks != curr.fsmblks) {
      LOG(INFO) << "[" << name << "] "
                << "Space in freed fastbin blocks (bytes): " << curr.fsmblks
                << ", diff: "
                << ((int64_t)curr.fsmblks - (int64_t)prev.fsmblks);
    }
    if (prev.uordblks != curr.uordblks) {
      LOG(INFO) << "[" << name << "] "
                << "Total allocated space (bytes): " << curr.uordblks
                << ", diff: "
                << ((int64_t)curr.uordblks - (int64_t)prev.uordblks);
    }
    if (prev.fordblks != curr.fordblks) {
      LOG(INFO) << "[" << name << "] "
                << "Total free space (bytes): " << curr.fordblks << ", diff: "
                << ((int64_t)curr.fordblks - (int64_t)prev.fordblks);
    }
    if (prev.keepcost != curr.keepcost) {
      LOG(INFO) << "[" << name << "] "
                << "Top-most, releasable space (bytes): " << curr.keepcost
                << ", diff: "
                << ((int64_t)curr.keepcost - (int64_t)prev.keepcost);
    }
    return curr;
  }
};

#define MACE_MEMORY_LOGGING_GUARD()                                        \
  MallinfoChangeLogger mem_logger_##__line__(std::string(__FILE__) + ":" + \
                                             std::string(__func__));
#else
#define MACE_MEMORY_LOGGING_GUARD()
#endif

}  // namespace mace

#endif  // MACE_UTILS_MEMORY_LOGGING_H_
