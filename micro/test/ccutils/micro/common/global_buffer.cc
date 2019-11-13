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

#include "micro/common/global_buffer.h"

#include "micro/base/logging.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace common {
namespace test {

namespace {
// for N=1, H=128, W=128, C=4, INPUT1&INPUT2&OUTPUT, sizeof(float)
const uint32_t kGlobalBufferSize = 128 * 128 * 4 * 3 * 4;
uint8_t kGlobalBuffer[kGlobalBufferSize];
GlobalBuffer global_buffer;
}

GlobalBuffer::GlobalBuffer() : offset_(0) {}
GlobalBuffer::~GlobalBuffer() {}

void GlobalBuffer::reset() {
  offset_ = 0;
}

void *GlobalBuffer::DoGetBuffer(uint32_t size) {
  if (size % 4 != 0) {
    size = (size + 3) / 4 * 4;
  }
  if (offset_ + size > kGlobalBufferSize) {
    LOG(FATAL) << "Global buffer is not enough."
               << "offset_: " << offset_ << ", size: " << size
               << ", kGlobalBufferSize: " << kGlobalBufferSize;
  }

  void *ptr = kGlobalBuffer + offset_;
  offset_ += size;

  return ptr;
}

GlobalBuffer *GetGlobalBuffer() {
  return &global_buffer;
}

}  // namespace test
}  // namespace common
}  // namespace micro
