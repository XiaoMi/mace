// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RPCMEMS_MTK_MTK_RPCMEM_H_
#define MACE_RPCMEMS_MTK_MTK_RPCMEM_H_

#include "mace/core/memory/rpcmem/rpcmem.h"

#include <unordered_map>

#include "mace/rpcmems/mtk/mtk_ion_wrapper.h"


namespace mace {
class MtkRpcmem : public Rpcmem {
 public:
  MtkRpcmem();
  virtual ~MtkRpcmem();

  void *New(int nbytes) override;
  void Delete(void *data) override;
  int ToFd(void *data) override;
  int SyncCacheStart(void *data) override;
  int SyncCacheEnd(void *data) override;

  RpcmemType GetRpcmemType() override;

 private:
  int ion_handle_;
  std::unique_ptr<MtkIonWrapper> mtk_ion_wrapper_;

  std::unordered_map<void *, int> addr_handle_map_;
  std::unordered_map<int, void *> handle_addr_map_;
  std::unordered_map<void *, int> addr_length_map_;
};

}  // namespace mace
#endif  // MACE_RPCMEMS_MTK_MTK_RPCMEM_H_
