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

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include "mace/core/file_storage.h"
#include "mace/utils/logging.h"

namespace mace {

std::shared_ptr<KVStorageFactory> kStorageFactory = nullptr;

FileStorage::FileStorage(const std::string &file_path):
    data_changed_(false), file_path_(file_path) {}

int FileStorage::Load() {
  struct stat st;
  if (stat(file_path_.c_str(), &st) == -1) {
    if (errno == ENOENT) {
      VLOG(1) << "File " << file_path_
              << " does not exist";
      return 0;
    } else {
      LOG(WARNING) << "Stat file " << file_path_
                   << " failed, error code: " << strerror(errno);
      return -1;
    }
  }
  utils::WriteLock lock(&data_mutex_);
  int fd = open(file_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    if (errno == ENOENT) {
      LOG(INFO) << "File " << file_path_
                << " does not exist";
      return 0;
    } else {
      LOG(WARNING) << "open file " << file_path_
                   << " failed, error code: " << strerror(errno);
      return -1;
    }
  }
  size_t file_size = st.st_size;
  unsigned char *file_data =
    static_cast<unsigned char *>(mmap(nullptr, file_size, PROT_READ,
          MAP_PRIVATE, fd, 0));
  int res = 0;
  if (file_data == MAP_FAILED) {
    LOG(WARNING) << "mmap file " << file_path_
                 << " failed, error code: " << strerror(errno);

    res = close(fd);
    if (res != 0) {
      LOG(WARNING) << "close file " << file_path_
                   << " failed, error code: " << strerror(errno);
    }
    return -1;
  }
  unsigned char *file_data_ptr = file_data;

  const size_t int_size = sizeof(int32_t);

  int64_t data_size = 0;
  memcpy(&data_size, file_data_ptr, sizeof(int64_t));
  file_data_ptr += sizeof(int64_t);
  int32_t key_size = 0;
  int32_t value_size = 0;
  for (int i = 0; i < data_size; ++i) {
    memcpy(&key_size, file_data_ptr, int_size);
    file_data_ptr += int_size;
    std::unique_ptr<char[]> key(new char[key_size+1]);
    memcpy(&key[0], file_data_ptr, key_size);
    file_data_ptr += key_size;
    key[key_size] = '\0';

    memcpy(&value_size, file_data_ptr, int_size);
    file_data_ptr += int_size;
    std::vector<unsigned char> value(value_size);
    memcpy(value.data(), file_data_ptr, value_size);
    file_data_ptr += value_size;

    data_.emplace(std::string(&key[0]), value);
  }

  res = munmap(file_data, file_size);
  if (res != 0) {
    LOG(WARNING) << "munmap file " << file_path_
                 << " failed, error code: " << strerror(errno);
    res = close(fd);
    if (res != 0) {
      LOG(WARNING) << "close file " << file_path_
                   << " failed, error code: " << strerror(errno);
    }
    return -1;
  }
  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }
  return 0;
}

void FileStorage::Clear() {
  utils::WriteLock lock(&data_mutex_);
  data_.clear();
  data_changed_ = true;
}

bool FileStorage::Insert(const std::string &key,
                         const std::vector<unsigned char> &value) {
  utils::WriteLock lock(&data_mutex_);
  auto res = data_.emplace(key, value);
  if (!res.second) {
    data_[key] = value;
  }
  data_changed_ = true;
  return true;
}

const std::vector<unsigned char> *FileStorage::Find(const std::string &key) {
  utils::ReadLock lock(&data_mutex_);
  auto iter = data_.find(key);
  if (iter == data_.end()) return nullptr;

  return &(iter->second);
}

int FileStorage::Flush() {
  utils::WriteLock lock(&data_mutex_);
  if (!data_changed_)  return 0;
  int fd = open(file_path_.c_str(), O_WRONLY | O_CREAT, 0600);
  if (fd < 0) {
    LOG(WARNING) << "open file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }

  const size_t int_size = sizeof(int32_t);

  int64_t data_size = sizeof(int64_t);
  for (auto &kv : data_) {
    data_size += 2 * int_size + kv.first.size() + kv.second.size();
  }
  std::unique_ptr<unsigned char[]> buffer(new unsigned char[data_size]);
  unsigned char *buffer_ptr = &buffer[0];

  int64_t num_of_data = data_.size();
  memcpy(buffer_ptr, &num_of_data, sizeof(int64_t));
  buffer_ptr += sizeof(int64_t);
  for (auto &kv : data_) {
    int32_t key_size = kv.first.size();
    memcpy(buffer_ptr, &key_size, int_size);
    buffer_ptr += int_size;

    memcpy(buffer_ptr, kv.first.c_str(), kv.first.size());
    buffer_ptr += kv.first.size();

    int32_t value_size = kv.second.size();
    memcpy(buffer_ptr, &value_size, int_size);
    buffer_ptr += int_size;

    memcpy(buffer_ptr, kv.second.data(), kv.second.size());
    buffer_ptr += kv.second.size();
  }
  int res = 0;
  buffer_ptr = &buffer[0];
  int64_t remain_size = data_size;
  while (remain_size > 0) {
    size_t buffer_size = std::min<int64_t>(remain_size, SSIZE_MAX);
    res = write(fd, buffer_ptr, buffer_size);
    if (res == -1) {
      LOG(WARNING) << "write file " << file_path_
                   << " failed, error code: " << strerror(errno);
      res = close(fd);
      if (res != 0) {
        LOG(WARNING) << "close file " << file_path_
                     << " failed, error code: " << strerror(errno);
      }
      return -1;
    }
    remain_size -= buffer_size;
    buffer_ptr += buffer_size;
  }

  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }
  data_changed_ = false;
  return 0;
}

};  // namespace mace
