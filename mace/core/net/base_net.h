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

#ifndef MACE_CORE_NET_BASE_NET_H_
#define MACE_CORE_NET_BASE_NET_H_

#include "mace/public/mace.h"
#include "mace/utils/macros.h"

namespace mace {

enum NetType {
  NT_CPU,
  NT_OPENCL,
  NT_HEXAGON,
  NT_HTA,
  NT_APU,
};

enum NetSubType {
  NT_SUB_REF,
};

class RunMetadata;

class BaseNet {
 public:
  BaseNet() noexcept = default;
  virtual ~BaseNet() = default;

  virtual MaceStatus Init() = 0;

  virtual MaceStatus Run(RunMetadata *run_metadata = nullptr) = 0;

  virtual MaceStatus AllocateIntermediateBuffer() = 0;

 protected:
  MACE_DISABLE_COPY_AND_ASSIGN(BaseNet);
};

}  // namespace mace

#endif  // MACE_CORE_NET_BASE_NET_H_
