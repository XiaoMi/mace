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

#include "mace/port/posix/file_system.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "mace/utils/memory.h"

namespace mace {
namespace port {

namespace {
class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion() = delete;
  PosixReadOnlyMemoryRegion(const void* addr, uint64_t length)
    : addr_(addr), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    if (length_ > 0) {
      munmap(const_cast<void *>(addr_), length_);
    }
  }
  const void *data() const override { return addr_; }
  uint64_t length() const override { return length_; }

 private:
  const void *addr_;
  const uint64_t length_;
};
}  // namespace

MaceStatus PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const char *fname,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  MaceStatus s = MaceStatus::MACE_SUCCESS;
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    // TODO(heliangliang) check errno
    s = MaceStatus::MACE_RUNTIME_ERROR;
  } else {
    struct stat st;
    fstat(fd, &st);
    if (st.st_size > 0) {
      const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (address == MAP_FAILED) {
        // TODO(heliangliang) check errno
        s = MaceStatus::MACE_RUNTIME_ERROR;
      } else {
        *result = make_unique<PosixReadOnlyMemoryRegion>(address, st.st_size);
      }
      close(fd);
    } else {
      // Empty file: mmap returns EINVAL (since Linux 2.6.12) length was 0
      *result = make_unique<PosixReadOnlyMemoryRegion>(nullptr, 0);
    }
  }
  return s;
}

}  // namespace port
}  // namespace mace
