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

#ifndef MACE_CORE_OPS_OP_INIT_CONTEXT_H_
#define MACE_CORE_OPS_OP_INIT_CONTEXT_H_

namespace mace {
class Workspace;
class Device;

// memory_optimizer, device
class OpInitContext {
 public:
  explicit OpInitContext(Workspace *ws, Device *device = nullptr);
  ~OpInitContext() = default;

  Workspace *workspace() const {
    return ws_;
  }

  void set_device(Device *device) {
    device_ = device;
  }

  Device *device() const {
    return device_;
  }

 private:
  Workspace *ws_;
  Device *device_;
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OP_INIT_CONTEXT_H_
