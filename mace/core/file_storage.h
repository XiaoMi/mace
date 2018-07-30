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

#ifndef MACE_CORE_FILE_STORAGE_H_
#define MACE_CORE_FILE_STORAGE_H_

#include <map>
#include <string>
#include <vector>

#include "mace/public/mace_runtime.h"
#include "mace/utils/rwlock.h"

namespace mace {

class FileStorage : public KVStorage {
 public:
  explicit FileStorage(const std::string &file_path);

 public:
  int Load() override;
  void Clear() override;
  bool Insert(const std::string &key,
              const std::vector<unsigned char> &value) override;
  const std::vector<unsigned char> *Find(const std::string &key) override;
  int Flush() override;

 private:
  bool data_changed_;
  std::string file_path_;
  std::map<std::string, std::vector<unsigned char>> data_;
  utils::RWMutex data_mutex_;
};

}  // namespace mace

#endif  // MACE_CORE_FILE_STORAGE_H_
