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

#ifndef MACE_CORE_KV_STORAGE_H_
#define MACE_CORE_KV_STORAGE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/rwlock.h"

namespace mace {

class KVStorage {
 public:
  // return: 0 for success, -1 for error
  virtual int Load() = 0;
  virtual bool Clear() = 0;
  // insert or update the key-value.
  virtual bool Insert(const std::string &key,
                      const std::vector<unsigned char> &value) = 0;
  virtual const std::vector<unsigned char> *Find(const std::string &key) = 0;
  // return: 0 for success, -1 for error
  virtual int Flush() = 0;
  virtual ~KVStorage() {}
};

class KVStorageFactory {
 public:
  virtual std::shared_ptr<KVStorage> CreateStorage(const std::string &name) = 0;

  virtual ~KVStorageFactory() {}
};

class FileStorageFactory : public KVStorageFactory {
 public:
  // You have to make sure your APP have read and write permission of the path.
  explicit FileStorageFactory(const std::string &path);

  ~FileStorageFactory();

  std::shared_ptr<KVStorage> CreateStorage(const std::string &name) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class FileStorage : public KVStorage {
 public:
  explicit FileStorage(const std::string &file_path);

 public:
  int Load() override;
  bool Clear() override;
  bool Insert(const std::string &key,
              const std::vector<unsigned char> &value) override;
  const std::vector<unsigned char> *Find(const std::string &key) override;
  int Flush() override;

 private:
  bool loaded_;
  bool data_changed_;
  std::string file_path_;
  std::map<std::string, std::vector<unsigned char>> data_;
  utils::RWMutex data_mutex_;
};


class ReadOnlyByteStreamStorage : public KVStorage {
 public:
  // load data from byte stream
  explicit ReadOnlyByteStreamStorage(const unsigned char *byte_stream,
                                     size_t byte_stream_size);

 public:
  int Load() override;
  bool Clear() override;
  bool Insert(const std::string &key,
              const std::vector<unsigned char> &value) override;
  const std::vector<unsigned char> *Find(const std::string &key) override;
  int Flush() override;

 private:
  std::map<std::string, std::vector<unsigned char>> data_;
};

}  // namespace mace

#endif  // MACE_CORE_KV_STORAGE_H_
