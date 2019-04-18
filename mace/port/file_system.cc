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

#include "mace/port/file_system.h"

#include <cstdio>
#include <string>

#include "mace/utils/memory.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {

namespace {
class StdWritableFile : public WritableFile {
 public:
  StdWritableFile(const std::string& fname, FILE* f)
      : fname_(fname), file_(f) {}

  ~StdWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  MaceStatus Append(const char *data, size_t length) override {
    size_t r = fwrite(data, 1, length, file_);
    if (r != length) {
      LOG(ERROR) << "Failed to append file: " << fname_
                 << ", requested: " << length
                 << ", written: " << r
                 << ", error: " << errno;
      return MaceStatus::MACE_RUNTIME_ERROR;
    }
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Close() override {
    if (fclose(file_) != 0) {
      LOG(ERROR) << "Failed to close file: " << fname_
                 << ", error: " << errno;
      return MaceStatus::MACE_RUNTIME_ERROR;
    }
    file_ = nullptr;
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Flush() override {
    if (fflush(file_) != 0) {
      LOG(ERROR) << "Failed to flush file: " << fname_
                 << ", error: " << errno;
      return MaceStatus::MACE_RUNTIME_ERROR;
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::string fname_;
  FILE* file_;
};

}  // namespace

WritableFile::~WritableFile() {}

MaceStatus FileSystem::NewWritableFile(const char *fname,
    std::unique_ptr<WritableFile>* result) {
  FILE* f = fopen(fname, "w");
  if (f == nullptr) {
      LOG(ERROR) << "Failed to open file to write: " << fname
                 << ", error: " << errno;
      return MaceStatus::MACE_RUNTIME_ERROR;
  } else {
    *result = make_unique<StdWritableFile>(fname, f);
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace port
}  // namespace mace
