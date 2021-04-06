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

#include "mace/core/memory/rpcmem/rpcmem.h"

#include "mace/utils/logging.h"

namespace mace {

Rpcmem::Rpcmem() : valid_detected_(false), valid_(false) {}

bool Rpcmem::IsRpcmemSupported() {
  if (!valid_detected_) {
    auto *ptr = New(1);
    valid_ = (ptr != nullptr);
    Delete(ptr);
    valid_detected_ = true;
    if (!valid_) {
      LOG(WARNING) << "Rpcmem is unsupported. type: " << GetRpcmemType();
    } else {
      LOG(INFO) << "Rpcmem is supported. type: " << GetRpcmemType();
    }
  }

  return valid_;
}

// If rpcmem is disabled, define empty functions.
#ifndef MACE_ENABLE_RPCMEM
std::shared_ptr<Rpcmem> CreateRpcmem(RpcmemType type) {
  MACE_UNUSED(type);
  return nullptr;
}
std::shared_ptr<Rpcmem> CreateRpcmem() {
  return nullptr;
}
#endif  // MACE_ENABLE_RPCMEM

}  // namespace mace
