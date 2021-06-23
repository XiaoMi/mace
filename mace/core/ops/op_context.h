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

#ifndef MACE_CORE_OPS_OP_CONTEXT_H_
#define MACE_CORE_OPS_OP_CONTEXT_H_

#include "mace/core/future.h"
#include "mace/core/runtime/runtime.h"
#include "mace/core/workspace.h"

namespace mace {

class OpContext {
 public:
  OpContext(Workspace *ws, Runtime *runtime);
  ~OpContext();
  void set_runtime(Runtime *runtime);
  Runtime *runtime() const;
  Workspace *workspace() const;

  void set_future(StatsFuture *future);
  StatsFuture *future() const;
  void set_fake_warmup(bool fake_warmup);
  bool fake_warmup() const;
 private:
  Runtime *runtime_;
  Workspace *ws_;
  StatsFuture *future_;
  bool fake_warmup_;
};

}  // namespace mace
#endif  // MACE_CORE_OPS_OP_CONTEXT_H_
