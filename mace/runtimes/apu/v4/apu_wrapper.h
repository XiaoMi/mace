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

#ifndef MACE_RUNTIMES_APU_V4_APU_WRAPPER_H_
#define MACE_RUNTIMES_APU_V4_APU_WRAPPER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>

#include "mace/runtimes/apu/v4/neuron_delegate_kernel.h"

#include "mace/core/quantize.h"
#include "mace/core/runtime/runtime.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"


namespace mace {

class ApuWrapper {
 public:
  explicit ApuWrapper(Runtime *runtime)
      :runtime_(runtime) {}
  bool Init(const NetDef &net_def, unsigned const char *model_data = nullptr,
            const char *file_name = nullptr,
            bool load = false, bool store = false);
  bool Run(const std::map<std::string, Tensor *> &input_tensors,
           std::map<std::string, Tensor *> *output_tensors);
  bool Uninit();

 private:
  neuron::NeuronDelegateKernel *frontend;
  bool AddOpsAndTensors(NetDef* net_def);
  Runtime *runtime_;
  bool initialised_ = false;

 private:
  // Access to NNApi.
  const NeuronApi* neuron_ = NeuronApiImplementation();
};

}  // namespace mace


#endif  // MACE_RUNTIMES_APU_V4_APU_WRAPPER_H_
