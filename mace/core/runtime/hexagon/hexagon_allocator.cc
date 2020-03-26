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

#include "mace/core/runtime/hexagon/hexagon_allocator.h"

namespace mace {

MaceStatus HexagonAllocator::New(size_t nbytes, void **result) {
  *result = rpcmem_.New(nbytes);
  MACE_CHECK_NOTNULL(*result);
  memset(*result, 0, nbytes);
  return MaceStatus::MACE_SUCCESS;
}

void HexagonAllocator::Delete(void *data) {
  rpcmem_.Delete(data);
}

bool HexagonAllocator::OnHost() const {
    return true;
}

Rpcmem *HexagonAllocator::rpcmem() {
  return &rpcmem_;
}

}  // namespace mace
