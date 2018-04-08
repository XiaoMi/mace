//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_FILE_STORAGE_H_
#define MACE_CORE_FILE_STORAGE_H_

#include <map>
#include <string>
#include <vector>

#include "mace/public/mace_runtime.h"

namespace mace {

class FileStorage : public KVStorage {
 public:
  explicit FileStorage(const std::string &file_path);

 public:
  int Load() override;
  bool Insert(const std::string &key,
              const std::vector<unsigned char> &value) override;
  const std::vector<unsigned char> *Find(const std::string &key) override;
  int Flush() override;

 private:
  std::string file_path_;
  std::map<std::string, std::vector<unsigned char>> data_;
};

}  // namespace mace

#endif  // MACE_CORE_FILE_STORAGE_H_
