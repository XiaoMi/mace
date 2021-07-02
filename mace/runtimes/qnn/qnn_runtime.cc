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

#include "mace/runtimes/qnn/qnn_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/qnn/qnn_wrapper.h"

namespace mace {

QnnRuntime::QnnRuntime(RuntimeContext *runtime_context)
    : QnnBaseRuntime(runtime_context) {}

MaceStatus QnnRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                   const MemoryType mem_type) {
  MACE_CHECK(GetRuntimeType() == RuntimeType::RT_HTP);
  qnn_wrapper_ = make_unique<QnnWrapper>(this);
  return QnnBaseRuntime::Init(config_impl, mem_type);
}

RuntimeType QnnRuntime::GetRuntimeType() {
  return RuntimeType::RT_HTP;
}

void RegisterQnnRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_HTP,
                        RuntimeSubType::RT_SUB_REF, QnnRuntime);
}
}  // namespace mace
