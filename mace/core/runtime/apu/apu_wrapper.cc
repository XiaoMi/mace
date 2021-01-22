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

#include "mace/core/runtime/apu/apu_wrapper.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mace/core/runtime/apu/neuron_delegate_builder.h"
#include "mace/core/quantize.h"

namespace mace {

bool ApuWrapper::Init(const NetDef &net_def, unsigned const char *model_data,
                      const char *file_name, bool load, bool store) {
  frontend = new neuron::NeuronDelegateKernel(neuron_, device_);
  MACE_CHECK(!(load & store),
            "Should not load and store the model simultaneously.");
  bool ret = frontend->Init(&net_def, model_data, load)
          && frontend->Prepare(file_name, load, store);
  if (!ret) {
    LOG(ERROR) << "ApuWrapper init failed.";
  } else {
    LOG(INFO) << "ApuWrapper init successfully.";
  }
  return ret;
}

bool ApuWrapper::Run(const std::map<std::string, Tensor *> &input_tensors,
                     std::map<std::string, Tensor *> *output_tensors) {
  bool ret = frontend->Eval(input_tensors, output_tensors);
  if (!ret) {
    LOG(ERROR) << "ApuWrapper Run failed.";
  } else {
    LOG(INFO) << "ApuWrapper Run successfully.";
  }
  return ret;
}

bool ApuWrapper::Uninit() {
  delete frontend;
  frontend = nullptr;
  return true;
}

}  // namespace mace
