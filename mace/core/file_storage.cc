//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/file_storage.h"

#include <fstream>
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
  std::ifstream ifs(file_path_, std::ios::binary | std::ios::in);
  if (ifs.is_open()) {
    int64_t data_size = 0;
    ifs.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
    while (data_size--) {
      int32_t key_size = 0;
      ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
      std::string key(key_size, ' ');
      ifs.read(&key[0], key_size);

      int32_t value_size = 0;
      ifs.read(reinterpret_cast<char *>(&value_size),
               sizeof(value_size));

      std::vector<unsigned char> value(value_size);
      ifs.read(reinterpret_cast<char *>(value.data()),
               value_size);
      data_.emplace(key, value);
    }
    ifs.close();
  } else {
    LOG(INFO) << "No file to Read.";
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
  std::ofstream ofs(file_path_,
                    std::ios::binary | std::ios::out);
  if (ofs.is_open()) {
    int64_t data_size = data_.size();
    ofs.write(reinterpret_cast<const char *>(&data_size),
              sizeof(data_size));
    for (auto &kv : data_) {
      int32_t key_size = static_cast<int32_t>(kv.first.size());
      ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
      ofs.write(kv.first.c_str(), key_size);

      int32_t value_size = static_cast<int32_t>(kv.second.size());
      ofs.write(reinterpret_cast<const char *>(&value_size),
                sizeof(value_size));
      ofs.write(reinterpret_cast<const char*>(kv.second.data()),
                value_size);
    }
    ofs.close();
  } else {
    LOG(WARNING) << "Write failed, please check directory exists";
  }
}

};  // namespace mace
