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

#include "mace/core/rpcmem.h"

#include "mace/utils/logging.h"

namespace mace {

Rpcmem::Rpcmem() {
  rpcmem_init(&rm);
  MACE_CHECK(rm.flag == 0, "rpcmem_init failed!");
}

Rpcmem::~Rpcmem() {
  rpcmem_deinit(&rm);
}

void *Rpcmem::New(int heapid, uint32_t flags, int nbytes) {
  return rpcmem_alloc(&rm, heapid, flags, nbytes);
}

void *Rpcmem::New(int nbytes) {
  return New(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_CACHED, nbytes);
}

void Rpcmem::Delete(void *data) {
  rpcmem_free(&rm, data);
}

int Rpcmem::ToFd(void *data) {
  return rpcmem_to_fd(&rm, data);
}

int Rpcmem::SyncCacheStart(void *data) {
  return rpcmem_sync_cache(&rm, data, RPCMEM_SYNC_START);
}

int Rpcmem::SyncCacheEnd(void *data) {
  return rpcmem_sync_cache(&rm, data, RPCMEM_SYNC_END);
}

}  // namespace mace
