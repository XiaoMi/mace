// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "micro/framework/scratch_buffer.h"

#include "micro/base/logging.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace framework {

#ifndef NDEBUG
namespace {
int64_t kDetectHandle = -1;
}
#endif

ScratchBuffer::ScratchBuffer(MaceMicroEngineConfig *engine_config) :
    engine_config_(engine_config), offset_(0) {
#ifndef NDEBUG
  int64_t cur_handle = reinterpret_cast<int64_t>(engine_config);
  MACE_ASSERT1(cur_handle != kDetectHandle, "Detect scratch buffer error.");
  kDetectHandle = cur_handle;
#endif
}

ScratchBuffer::~ScratchBuffer() {
#ifndef NDEBUG
  kDetectHandle = -1;
#endif
}

void *ScratchBuffer::DoGetBuffer(uint32_t size) {
  if (size % 4 != 0) {
    size = (size + 3) / 4 * 4;
  }
  if (offset_ + size > engine_config_->scratch_buffer_size_) {
    LOG(FATAL) << "The scratch buffer is not enough."
               << "offset_: " << offset_ << ", size: " << size
               << ", engine_config_->scratch_buffer_size_: "
               << engine_config_->scratch_buffer_size_;
  }

  void *ptr = engine_config_->scratch_buffer_ + offset_;
  offset_ += size;

  return ptr;
}

}  // namespace framework
}  // namespace micro
