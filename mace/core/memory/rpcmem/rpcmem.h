// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_MEMORY_RPCMEM_RPCMEM_H_
#define MACE_CORE_MEMORY_RPCMEM_RPCMEM_H_

#include <memory>
#include <cstdint>

namespace mace {

enum RpcmemType {
  ION_QUALCOMM = 0,
  ION_MTK,

  ION_TYPE_NUM,  // The number of rpcmem type
};

class Rpcmem {
 public:
  Rpcmem();
  virtual ~Rpcmem() = default;

  bool IsRpcmemSupported();

  virtual void *New(int heapid, uint32_t flags, int nbytes) = 0;
  virtual void *New(int nbytes) = 0;
  virtual void Delete(void *data) = 0;
  virtual int GetDefaultHeapId() = 0;
  virtual int ToFd(void *data) = 0;
  virtual int SyncCacheStart(void *data) = 0;
  virtual int SyncCacheEnd(void *data) = 0;

  virtual int GetIonCacheFlag() = 0;
  virtual RpcmemType GetRpcmemType() = 0;

 protected:
  bool valid_detected_;
  bool valid_;
};

namespace rpcmem_factory {
extern std::shared_ptr<Rpcmem> CreateRpcmem(RpcmemType type);
extern std::shared_ptr<Rpcmem> CreateRpcmem();
}  // namespace rpcmem_factory

}  // namespace mace
#endif  // MACE_CORE_MEMORY_RPCMEM_RPCMEM_H_
