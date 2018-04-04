//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/file_storage_engine.h"

#include <fstream>

#include "mace/utils/logging.h"

namespace mace {

std::string FileStorageEngine::kStoragePath  // NOLINT(runtime/string)
    = "/data/local/tmp";

FileStorageEngine::FileStorageEngine(const std::string &file_name):
    file_name_(file_name) {}

void FileStorageEngine::Write(
    const std::map<std::string, std::vector<unsigned char>> &data) {
  const std::string file_path = kStoragePath + "/" + file_name_;

  std::ofstream ofs(file_path,
                    std::ios::binary | std::ios::out);
  if (ofs.is_open()) {
    int64_t data_size = data.size();
    ofs.write(reinterpret_cast<const char *>(&data_size),
              sizeof(data_size));
    for (auto &kv : data) {
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

void FileStorageEngine::Read(
    std::map<std::string, std::vector<unsigned char>> *data) {
  const std::string file_path = kStoragePath + "/" + file_name_;
  std::ifstream ifs(file_path, std::ios::binary | std::ios::in);
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

      std::vector<unsigned char> program_binary(value_size);
      ifs.read(reinterpret_cast<char *>(program_binary.data()),
               value_size);
      data->emplace(key, program_binary);
    }
    ifs.close();
  } else {
    LOG(INFO) << "No file to Read.";
  }
}

};  // namespace mace
