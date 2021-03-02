//  Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/core/net_optimizer.h"

#include <string>

namespace mace {

RuntimeType NetOptimizer::SelectBestRuntime(
    const OperatorDef *op_def,
    RuntimeType target_runtime_type,
    const std::set<RuntimeType> &available_runtimes,
    const std::vector<RuntimeType> &inputs_op_runtimes) {
  static const std::set<std::string> kComputeIntensiveOps = {
      "Conv2D", "DepthwiseConv2d", "Deconv2D", "DepthwiseDeconv2d",
      "FullyConnected"
  };
  // CPU is the runtime to fall back
  RuntimeType best_runtime = RuntimeType::RT_CPU;
  if (available_runtimes.count(target_runtime_type) == 1) {
    best_runtime = target_runtime_type;
  }
  if (best_runtime == RuntimeType::RT_CPU) {
    return best_runtime;
  }
  // Put compute-intensive ops in target runtime
  if (kComputeIntensiveOps.count(op_def->type()) == 1) {
    return best_runtime;
  }
  // Greedy strategy: Use input op's device type as current op's device
  for (auto runtime_type : inputs_op_runtimes) {
    if (runtime_type == best_runtime) {
      return best_runtime;
    }
  }
  return RuntimeType::RT_CPU;
}
}  // namespace mace
