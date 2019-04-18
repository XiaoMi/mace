// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <climits>
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include "mace/core/kv_storage.h"
#include "mace/utils/macros.h"
#include "mace/utils/logging.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"

namespace mace {

namespace {
void ParseKVData(const unsigned char *data,
                 size_t data_size,
                 std::map<std::string, std::vector<unsigned char>> *kv_map) {
  const size_t int_size = sizeof(int32_t);

  size_t parsed_offset = 0;
  int64_t num_tuple = 0;
  memcpy(&num_tuple, data, sizeof(num_tuple));
  data += sizeof(num_tuple);
  parsed_offset += sizeof(num_tuple);
  int32_t key_size = 0;
  int32_t value_size = 0;
  for (int i = 0; i < num_tuple; ++i) {
    memcpy(&key_size, data, int_size);
    data += int_size;
    std::unique_ptr<char[]> key(new char[key_size+1]);
    memcpy(&key[0], data, key_size);
    data += key_size;
    key[key_size] = '\0';
    parsed_offset += int_size + key_size;

    memcpy(&value_size, data, int_size);
    data += int_size;
    std::vector<unsigned char> value(value_size);
    memcpy(value.data(), data, value_size);
    data += value_size;
    parsed_offset += int_size + value_size;
    MACE_CHECK(parsed_offset <= data_size,
               "Paring storage data out of range: ",
               parsed_offset, " > ", data_size);

    kv_map->emplace(std::string(&key[0]), value);
  }
}

}  // namespace

class FileStorageFactory::Impl {
 public:
  explicit Impl(const std::string &path);

  std::shared_ptr<KVStorage> CreateStorage(const std::string &name);

 private:
  std::string path_;
};

FileStorageFactory::Impl::Impl(const std::string &path): path_(path) {}

std::shared_ptr<KVStorage> FileStorageFactory::Impl::CreateStorage(
    const std::string &name) {
  return std::shared_ptr<KVStorage>(new FileStorage(path_ + "/" + name));
}

FileStorageFactory::FileStorageFactory(const std::string &path):
    impl_(new FileStorageFactory::Impl(path)) {}

FileStorageFactory::~FileStorageFactory() = default;

std::shared_ptr<KVStorage> FileStorageFactory::CreateStorage(
    const std::string &name) {
  return impl_->CreateStorage(name);
}

FileStorage::FileStorage(const std::string &file_path):
    loaded_(false), data_changed_(false), file_path_(file_path) {}

int FileStorage::Load() {
  utils::WriteLock lock(&data_mutex_);
  if (loaded_) {
    return 0;
  }

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> kv_data =
    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  auto fs = GetFileSystem();
  auto status = fs->NewReadOnlyMemoryRegionFromFile(
      file_path_.c_str(), &kv_data);
  if (status != MaceStatus::MACE_SUCCESS)  {
    LOG(ERROR) << "Failed to read kv store file: " << file_path_;
    return -1;
  } else {
    ParseKVData(static_cast<const unsigned char *>(kv_data->data()),
        kv_data->length(), &data_);
  }

  loaded_ = true;
  return 0;
}

bool FileStorage::Clear() {
  utils::WriteLock lock(&data_mutex_);
  if (!data_.empty()) {
    data_.clear();
    data_changed_ = true;
  }
  return true;
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
  auto fs = GetFileSystem();
  std::unique_ptr<port::WritableFile> file;
  MaceStatus s = fs->NewWritableFile(file_path_.c_str(), &file);
  if (s != MaceStatus::MACE_SUCCESS) {
    LOG(WARNING) << "open file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }

  const size_t int_size = sizeof(int32_t);

  int64_t data_size = sizeof(int64_t);
  for (auto &kv : data_) {
    data_size += 2 * int_size + kv.first.size() + kv.second.size();
  }
  std::unique_ptr<char[]> buffer = make_unique<char[]>(data_size);
  char *buffer_ptr = buffer.get();

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
  s = file->Append(buffer.get(), data_size);
  if (s != MaceStatus::MACE_SUCCESS) {
    file->Close();
    return -1;
  }
  s = file->Close();
  if (s != MaceStatus::MACE_SUCCESS) {
    return -1;
  }

  data_changed_ = false;
  return 0;
}


ReadOnlyByteStreamStorage::ReadOnlyByteStreamStorage(
    const unsigned char *byte_stream, size_t byte_stream_size) {
  ParseKVData(byte_stream, byte_stream_size, &data_);
}

int ReadOnlyByteStreamStorage::Load() {
  return 0;
}

bool ReadOnlyByteStreamStorage::Clear() {
  LOG(FATAL) << "ReadOnlyByteStreamStorage should not clear data";
  return true;
}

const std::vector<unsigned char>* ReadOnlyByteStreamStorage::Find(
    const std::string &key) {
  auto iter = data_.find(key);
  if (iter == data_.end()) return nullptr;

  return &(iter->second);
}

bool ReadOnlyByteStreamStorage::Insert(
    const std::string &key,
    const std::vector<unsigned char> &value) {
  MACE_UNUSED(key);
  MACE_UNUSED(value);
  LOG(FATAL) << "ReadOnlyByteStreamStorage should not insert data";
  return true;
}

int ReadOnlyByteStreamStorage::Flush() {
  return 0;
}

};  // namespace mace
