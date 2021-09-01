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

#include "mace/utils/logging.h"

#if defined(__aarch64__) && defined(__ANDROID__)
#include "mace/rpcmems/mtk/mtk_rpcmem.h"
#endif
#include "mace/rpcmems/dmabufheap/dma_buf_heap_rpcmem.h"
#include "mace/rpcmems/qualcomm/qualcomm_rpcmem.h"

namespace mace {
namespace rpcmem_factory {

std::shared_ptr<Rpcmem> CreateRpcmem(RpcmemType type) {
  switch (type) {
    case ION_QUALCOMM: {
      return std::make_shared<QualcommRpcmem>();
    }
    case ION_MTK: {
#if defined(__aarch64__) && defined(__ANDROID__)
      return std::make_shared<MtkRpcmem>();
#else
      return nullptr;
#endif
    }
    case DMA_BUF_HEAP: {
      return std::make_shared<DmaBufHeapRpcmem>();
    }
    default: {
      LOG(FATAL) << "Invalid RpcmemType: " << type;
      return nullptr;
    }
  }
}

std::shared_ptr<Rpcmem> CreateRpcmem() {
  for (int i = 0; i < RpcmemType::ION_TYPE_NUM; ++i) {
    auto type = static_cast<RpcmemType>(i);
    auto rpcmem = CreateRpcmem(type);
    if (rpcmem != nullptr && rpcmem->IsRpcmemSupported()) {
      LOG(INFO) << "Rpcmem is supported. type: " << i;
      return rpcmem;
    }
  }

  return nullptr;
}

}  // namespace rpcmem_factory
}  // namespace mace
