//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/file_storage.h"

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <utility>


#include "mace/utils/logging.h"

namespace mace {

class FileStorageFactory::Impl {
 public:
  explicit Impl(const std::string &path);

  std::unique_ptr<KVStorage> CreateStorage(const std::string &name);

 private:
  std::string path_;
};

FileStorageFactory::Impl::Impl(const std::string &path): path_(path) {}
std::unique_ptr<KVStorage> FileStorageFactory::Impl::CreateStorage(
    const std::string &name) {
  return std::move(std::unique_ptr<KVStorage>(
      new FileStorage(path_ + "/" + name)));
}

FileStorageFactory::FileStorageFactory(const std::string &path):
    impl_(new FileStorageFactory::Impl(path)) {}

FileStorageFactory::~FileStorageFactory() = default;

std::unique_ptr<KVStorage> FileStorageFactory::CreateStorage(
    const std::string &name) {
  return impl_->CreateStorage(name);
}

FileStorage::FileStorage(const std::string &file_path):
    file_path_(file_path) {}

void FileStorage::Load() {
  struct stat st;
  stat(file_path_.c_str(), &st);
  size_t file_size = st.st_size;
  int fd = open(file_path_.c_str(), O_RDONLY);
  if (fd == -1) {
    LOG(WARNING) << "open file " << file_path_
                 << " failed, error code: " << errno;
    return;
  }
  unsigned char *file_data =
    static_cast<unsigned char *>(mmap(nullptr, file_size, PROT_READ,
          MAP_PRIVATE, fd, 0));
  int res = 0;
  if (file_data == MAP_FAILED) {
    LOG(WARNING) << "mmap file " << file_path_
                 << " failed, error code: " << errno;

    res = close(fd);
    if (res != 0) {
      LOG(WARNING) << "close file " << file_path_
                   << " failed, error code: " << errno;
    }
    return;
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
                 << " failed, error code: " << errno;
    return;
  }
  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << errno;
  }
}

bool FileStorage::Insert(const std::string &key,
                         const std::vector<unsigned char> &value) {
  data_.emplace(key, value);
  return true;
}

std::vector<unsigned char> *FileStorage::Find(const std::string &key) {
  auto iter = data_.find(key);
  if (iter == data_.end()) return nullptr;

  return &(iter->second);
}

void FileStorage::Flush() {
  int fd = open(file_path_.c_str(), O_WRONLY | O_CREAT, 0600);
  if (fd < 0) {
    LOG(WARNING) << "open file " << file_path_
                 << " failed, error code:" << errno;
    return;
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
                   << " failed, error code: " << errno;
      return;
    }
    remain_size -= buffer_size;
    buffer_ptr += buffer_size;
  }

  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << errno;
  }
}

};  // namespace mace
