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

#include "mace/port/windows/file_system.h"

#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/utils/memory.h"

namespace mace {
namespace port {

namespace {
class WindowsReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  WindowsReadOnlyMemoryRegion() = delete;
  explicit WindowsReadOnlyMemoryRegion(std::vector<char> &&buffer) {
     buffer_.swap(buffer);  // forward
    }
  ~WindowsReadOnlyMemoryRegion() override {}
  const void *data() const override { return buffer_.data(); }
  uint64_t length() const override { return buffer_.size(); }

 private:
  std::vector<char> buffer_;
};
}  // namespace

MaceStatus WindowsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const char *fname, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  // TODO(heliangliang) change to CreateFileMapping
  std::ifstream ifs(fname, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  ifs.seekg(0, ifs.end);
  size_t length = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  std::vector<char> buffer(length);
  ifs.read(reinterpret_cast<char *>(buffer.data()), length);

  if (ifs.fail()) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  ifs.close();
  *result = make_unique<WindowsReadOnlyMemoryRegion>(std::move(buffer));
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace port
}  // namespace mace
