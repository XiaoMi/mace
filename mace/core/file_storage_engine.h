//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_FILE_STORAGE_ENGINE_H_
#define MACE_CORE_FILE_STORAGE_ENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "mace/public/mace_runtime.h"

namespace mace {

class FileStorageEngine : public KVStorageEngine {
 public:
  explicit FileStorageEngine(const std::string &file_name);
 public:
  void Write(
      const std::map<std::string, std::vector<unsigned char>> &data) override;
  void Read(
      std::map<std::string, std::vector<unsigned char>> *data) override;
 private:
  std::string file_name_;

 public:
  static std::string kStoragePath;
};

}  // namespace mace

#endif  // MACE_CORE_FILE_STORAGE_ENGINE_H_
