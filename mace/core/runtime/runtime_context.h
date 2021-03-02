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


#ifndef MACE_CORE_RUNTIME_RUNTIME_CONTEXT_H_
#define MACE_CORE_RUNTIME_RUNTIME_CONTEXT_H_

#ifdef MACE_ENABLE_RPCMEM
#include "mace/core/memory/rpcmem/rpcmem.h"
#endif  // MACE_ENABLE_RPCMEM

#include "mace/utils/thread_pool.h"

namespace mace {

enum RuntimeContextType {
  RCT_NORMAL,
  RCT_QC_ION,
};

class RuntimeContext {
 public:
  RuntimeContext(utils::ThreadPool *thrd_pool)
      : context_type(RCT_NORMAL), thread_pool(thrd_pool) {}

  virtual ~RuntimeContext() = default;

 public:
  RuntimeContextType context_type;
  utils::ThreadPool *thread_pool;
};

#ifdef MACE_ENABLE_RPCMEM
class QcIonRuntimeContext : public RuntimeContext {
 public:
  QcIonRuntimeContext(utils::ThreadPool *thrd_pool, std::shared_ptr<Rpcmem> rm)
      : RuntimeContext(thrd_pool), rpcmem(rm) {
    context_type = RCT_QC_ION;
  }

  ~QcIonRuntimeContext() = default;

 public:
  std::shared_ptr<Rpcmem> rpcmem;
};
#endif  // MACE_ENABLE_RPCMEM

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_RUNTIME_CONTEXT_H_
