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


#include "mace/flows/cpu/cpu_ref_flow.h"
#include "mace/flows/cpu/transpose_const.h"

#include "mace/core/flow/flow_registry.h"
#include "mace/core/net_def_adapter.h"
#include "mace/core/net/serial_net.h"
#include "mace/core/workspace.h"
#include "mace/proto/mace.pb.h"

namespace mace {

CpuRefFlow::CpuRefFlow(FlowContext *flow_context)
    : CommonFp32Flow(flow_context) {
  VLOG(3) << "CpuRefFlow::CpuRefFlow";
}

MaceStatus CpuRefFlow::Init(const NetDef *net_def,
                            const unsigned char *model_data,
                            const int64_t model_data_size,
                            bool *model_data_unused) {
  MACE_RETURN_IF_ERROR(BaseFlow::Init(net_def, model_data, model_data_size,
                                      model_data_unused));

  MACE_RETURN_IF_ERROR(ws_->LoadModelTensor(
      *net_def, main_runtime_, model_data, model_data_size));

  NetDef adapted_net_def;
  NetDefAdapter net_def_adapter(op_registry_, ws_.get());
  net_def_adapter.AdaptNetDef(net_def, main_runtime_,
                              cpu_runtime_, &adapted_net_def);
  if (!is_quantized_model_) {
    TransposeConstForCPU(&cpu_runtime_->thread_pool(), ws_.get(), cpu_runtime_,
                         &adapted_net_def);
  }
  // Init model
  net_ = std::unique_ptr<BaseNet>(new SerialNet(op_registry_,
                                                &adapted_net_def,
                                                ws_.get(),
                                                main_runtime_,
                                                cpu_runtime_));
  if (model_data_unused != nullptr) {
    *model_data_unused = ws_->diffused_buffer();
  }
  if (main_runtime_->GetRuntimeType() == RuntimeType::RT_OPENCL) {
    ws_->RemoveAndReloadBuffer(adapted_net_def, model_data, main_runtime_);
    if (model_data_unused != nullptr) {
      *model_data_unused = true;
    }
  }
  MACE_RETURN_IF_ERROR(net_->Init());
  MACE_RETURN_IF_ERROR(ws_->AddQuantizeInfoForOutputTensor(adapted_net_def,
                                                           main_runtime_));

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus CpuRefFlow::Run(TensorMap *input_tensors,
                           TensorMap *output_tensors,
                           RunMetadata *run_metadata) {
  VLOG(1) << "CpuRefFlow::Run";
  MACE_UNUSED(input_tensors);
  MACE_UNUSED(output_tensors);
  return net_->Run(run_metadata, false);
}

MaceStatus CpuRefFlow::GetInputTransposeDims(
    const std::pair<const std::string, MaceTensor> &input,
    const Tensor *input_tensor,
    std::vector<int> *dst_dims, DataFormat *data_format) {
  *data_format = input.second.data_format();
  if (input_tensor->data_format() != DataFormat::NONE) {
    if (!is_quantized_model_ && input.second.shape().size() == 4 &&
        input.second.data_format() == DataFormat::NHWC) {
      VLOG(1) << "Transform input " << input.first << " from NHWC to NCHW";
      *data_format = DataFormat::NCHW;
      *dst_dims = {0, 3, 1, 2};
    } else if (is_quantized_model_ &&
        input.second.data_format() == DataFormat::NCHW) {
      VLOG(1) << "Transform input " << input.first << " from NCHW to NHWC";
      *data_format = DataFormat::NHWC;
      *dst_dims = {0, 2, 3, 1};
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

void RegisterCpuRefFlow(FlowRegistry *flow_registry) {
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_CPU,
                     FlowSubType::FW_SUB_REF, CpuRefFlow);
}

}  // namespace mace
