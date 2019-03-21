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

#ifndef MACE_PORT_FILE_SYSTEM_H_
#define MACE_PORT_FILE_SYSTEM_H_

#include <string>
#include <memory>

#include "mace/public/mace.h"

namespace mace {
namespace port {

class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() = default;
  virtual ~ReadOnlyMemoryRegion() = default;
  virtual const void *data() = 0;
  virtual uint64_t length() = 0;
};

class FileSystem {
 public:
  FileSystem() = default;
  virtual ~FileSystem() = default;
  virtual MaceStatus NewReadOnlyMemoryRegionFromFile(const char *fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;
};

}  // namespace port
}  // namespace mace

#endif  // MACE_PORT_FILE_SYSTEM_H_
