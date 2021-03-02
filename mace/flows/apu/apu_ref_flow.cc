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


#include "mace/flows/apu/apu_ref_flow.h"

#include "mace/core/flow/flow_registry.h"
#include "mace/runtimes/apu/apu_runtime.h"

namespace mace {

ApuRefFlow::ApuRefFlow(FlowContext *flow_context)
    : CommonFp32Flow(flow_context) {}

MaceStatus ApuRefFlow::Init(const NetDef *net_def,
                            const unsigned char *model_data,
                            const int64_t model_data_size,
                            bool *model_data_unused) {
  if (model_data_unused != nullptr) {
    *model_data_unused = true;
  }

  auto succ = BaseFlow::Init(net_def, model_data,
                             model_data_size, model_data_unused);
  MACE_RETURN_IF_ERROR(succ);

  // create output tensor
  MACE_RETURN_IF_ERROR(InitOutputTensor());

  auto *apu_runtime = ApuRuntime::Get(main_runtime_);
  auto apu_cache_policy = apu_runtime->GetCachePolicy();
  bool cache_load = apu_cache_policy == APUCachePolicy::APU_CACHE_LOAD;
  bool cache_store = apu_cache_policy == APUCachePolicy::APU_CACHE_STORE;
  const char *file_name = cache_store ? apu_runtime->GetCacheStorePath()
                                      : apu_runtime->GetCacheLoadPath();
  auto *apu_wrapper = apu_runtime->GetApuWrapper();
  bool ret = false;
  if (cache_load || cache_store) {
    VLOG(1) << "Loading/Storing init cache";
    ret = apu_wrapper->Init(*net_def, model_data, file_name,
                            cache_load, cache_store);
  }
  if (!ret && !cache_store) {
    VLOG(1) << "Do not use init cache";
    ret = apu_wrapper->Init(*net_def, model_data);
  }
  MACE_CHECK(ret, "apu int error", cache_load, cache_store);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ApuRefFlow::Run(TensorMap *input_tensors,
                           TensorMap *output_tensors,
                           RunMetadata *run_metadata) {
  MACE_UNUSED(run_metadata);
  auto *apu_wrapper = ApuRuntime::Get(main_runtime_)->GetApuWrapper();
  MACE_CHECK(apu_wrapper->Run(*input_tensors, output_tensors),
             "apu run error");
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ApuRefFlow::GetInputTransposeDims(
    const std::pair<const std::string, MaceTensor> &input,
    const Tensor *input_tensor,
    std::vector<int> *dst_dims, DataFormat *data_format) {
  *data_format = input.second.data_format();
  if (input_tensor->data_format() != DataFormat::NONE) {
    if (is_quantized_model_ && input.second.shape().size() == 4 &&
        input.second.data_format() == DataFormat::NCHW) {
      VLOG(1) << "Transform input " << input.first << " from NCHW to NHWC";
      *data_format = DataFormat::NHWC;
      *dst_dims = {0, 2, 3, 1};
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

void RegisterApuRefFlow(FlowRegistry *flow_registry) {
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_APU,
                     FlowSubType::FW_SUB_REF, ApuRefFlow);
}

}  // namespace mace
