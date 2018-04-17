//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/public/mace_runtime.h"
#include "mace/utils/logging.h"

namespace mace {

std::shared_ptr<KVStorageFactory> kStorageFactory = nullptr;

void SetKVStorageFactory(std::shared_ptr<KVStorageFactory> storage_factory) {
  VLOG(1) << "Set internal KV Storage Engine";
  kStorageFactory = storage_factory;
}

};  // namespace mace
