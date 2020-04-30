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

#ifndef MACE_CORE_PROTO_NET_DEF_HELPER_H_
#define MACE_CORE_PROTO_NET_DEF_HELPER_H_

#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class NetDefHelper {
 public:
  static bool HasQuantizedTensor(const NetDef &net_def);
  static bool HasHalfTensor(const NetDef &net_def);
  static index_t GetModelValidSize(const NetDef &net_def);
  static bool IsQuantizedModel(const NetDef &net_def);
};

}  // namespace mace

#endif  // MACE_CORE_PROTO_NET_DEF_HELPER_H_
