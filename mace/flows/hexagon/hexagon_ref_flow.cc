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


#include "mace/flows/hexagon/hexagon_ref_flow.h"

#include "mace/core/flow/flow_registry.h"
#include "mace/runtimes/hexagon/hexagon_base_runtime.h"

namespace mace {

HexagonRefFlow::HexagonRefFlow(FlowContext *flow_context)
    : CommonFp32Flow(flow_context) {
  VLOG(3) << "HexagonRefFlow::HexagonRefFlow";
}

MaceStatus HexagonRefFlow::Init(const NetDef *net_def,
                                const unsigned char *model_data,
                                const int64_t model_data_size,
                                bool *model_data_unused) {
  if (model_data_unused != nullptr) {
    *model_data_unused = true;
  }

  auto ret = BaseFlow::Init(net_def, model_data,
                            model_data_size, model_data_unused);
  MACE_RETURN_IF_ERROR(ret);

  // create output tensor
  MACE_RETURN_IF_ERROR(InitOutputTensor());

  // Init hexagon net
  auto *hexagon_runtime = HexagonBaseRuntime::Get(main_runtime_);
  auto *hexagon_controller = hexagon_runtime->GetHexagonController();
  MACE_CHECK(hexagon_controller->SetupGraph(
      *net_def, model_data, model_data_size), "hexagon setup graph error");
  if (VLOG_IS_ON(2)) {
    hexagon_controller->PrintGraph();
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus HexagonRefFlow::Run(TensorMap *input_tensors,
                               TensorMap *output_tensors,
                               RunMetadata *run_metadata) {
  MACE_UNUSED(run_metadata);
  VLOG(1) << "HexagonRefFlow::Run";
  auto hexagon_runtime = HexagonBaseRuntime::Get(main_runtime_);
  bool ret = hexagon_runtime->ExecuteGraphNew(*input_tensors, output_tensors);

  return ret ? MaceStatus::MACE_SUCCESS : MaceStatus::MACE_RUNTIME_ERROR;
}

MaceStatus HexagonRefFlow::GetInputTransposeDims(
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

void RegisterHexagonRefFlow(FlowRegistry *flow_registry) {
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_HEXAGON,
                     FlowSubType::FW_SUB_REF, HexagonRefFlow);
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_HTA,
                     FlowSubType::FW_SUB_REF, HexagonRefFlow);
}

}  // namespace mace
