// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <random>
#include <string>

#include "mace/core/runtime_failure_mock.h"
#include "mace/utils/logging.h"

namespace mace {

namespace {
inline float GetRuntimeFailureRatioFromEnv() {
  const char *env = getenv("MACE_RUNTIME_FAILURE_RATIO");
  if (env == nullptr) {
    return 0;
  }
  std::string env_str(env);
  std::istringstream ss(env_str);
  float ratio;
  ss >> ratio;
  return ratio;
}
}  // namespace

bool ShouldMockRuntimeFailure() {
  static float mock_runtime_failure_ratio = GetRuntimeFailureRatioFromEnv();
  if (mock_runtime_failure_ratio > 1e-6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float random_ratio = dis(gen);
    if (random_ratio < mock_runtime_failure_ratio) {
      VLOG(0) << "Mock runtime failure.";
      return true;
    }
  }

  return false;
}

}  // namespace mace
