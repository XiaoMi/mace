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

#ifndef MICRO_TEST_CCUTILS_MICRO_COMMON_GLOBAL_BUFFER_H_
#define MICRO_TEST_CCUTILS_MICRO_COMMON_GLOBAL_BUFFER_H_

#include "micro/base/logging.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace common {
namespace test {

class GlobalBuffer {
 public:
  GlobalBuffer();
  ~GlobalBuffer();

  void reset();

  template<typename T>
  T *GetBuffer(int32_t size) {
    MACE_ASSERT(size > 0);
    return static_cast<T *>(
        DoGetBuffer(static_cast<uint32_t>(size) * sizeof(T)));
  }

  template<typename T>
  T *GetBuffer(uint32_t size) {
    return static_cast<T *>(DoGetBuffer(size * sizeof(T)));
  }

 private:
  void *DoGetBuffer(uint32_t size);

 private:
  uint32_t offset_;
};

GlobalBuffer *GetGlobalBuffer();

}  // namespace test
}  // namespace common
}  // namespace micro

#endif  // MICRO_TEST_CCUTILS_MICRO_COMMON_GLOBAL_BUFFER_H_
