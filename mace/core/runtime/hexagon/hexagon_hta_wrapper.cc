// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/core/runtime/hexagon/hexagon_hta_wrapper.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "mace/core/runtime/hexagon/hexagon_hta_ops.h"
#include "mace/core/types.h"
#include "mace/utils/memory.h"
#include "mace/core/quantize.h"
#include "third_party/hta/hta_hexagon_api.h"

namespace mace {

HexagonHTAWrapper::HexagonHTAWrapper(Device *device)
    : quantize_util_(&device->cpu_runtime()->thread_pool()) {
}

int HexagonHTAWrapper::GetVersion() {
  int version;
  MACE_CHECK(hexagon_hta_nn_version(&version) == 0, "get version error");
  return version;
}

bool HexagonHTAWrapper::Config() {
  LOG(INFO) << "HTA config";
  MACE_CHECK(hexagon_hta_nn_config() == 0, "hexagon config error");
  return true;
}

bool HexagonHTAWrapper::Init() {
  LOG(INFO) << "Hexagon init";
  MACE_CHECK(hexagon_hta_nn_init(&nn_id_) == 0, "hexagon_nn_init failed");
  ResetPerfInfo();
  return true;
}

bool HexagonHTAWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  return true;
}

bool HexagonHTAWrapper::SetupGraph(const NetDef &net_def,
                                   unsigned const char *model_data) {
  LOG(INFO) << "Hexagon setup graph";

  int64_t t0 = NowMicros();

  // const node
  for (const ConstTensor &const_tensor : net_def.tensors()) {
    std::vector<int> tensor_shape(const_tensor.dims().begin(),
                                  const_tensor.dims().end());
    while (tensor_shape.size() < 4) {
      tensor_shape.insert(tensor_shape.begin(), 1);
    }

    hexagon_nn_const_node const_node;
    const_node.node_id = node_id(const_tensor.node_id());
    const_node.tensor.batches = tensor_shape[0];
    const_node.tensor.height = tensor_shape[1];
    const_node.tensor.width = tensor_shape[2];
    const_node.tensor.depth = tensor_shape[3];

    if (const_tensor.data_type() == DataType::DT_INT32 &&
        const_tensor.data_size() == 0) {
      const_node.tensor.data = NULL;
      const_node.tensor.dataLen = 0;
    } else {
      const_node.tensor.data =
          const_cast<unsigned char *>(model_data + const_tensor.offset());
      const_node.tensor.dataLen = const_tensor.data_size() *
          GetEnumTypeSize(const_tensor.data_type());
    }

    hexagon_hta_nn_append_const_node(nn_id_,
                                     const_node.node_id,
                                     const_node.tensor.batches,
                                     const_node.tensor.height,
                                     const_node.tensor.width,
                                     const_node.tensor.depth,
                                     const_node.tensor.data,
                                     const_node.tensor.dataLen);
  }

  // op node
  OpMap op_map;
  op_map.Init();
  std::vector<std::vector<hexagon_hta_nn_input>> cached_inputs;
  std::vector<std::vector<hexagon_hta_nn_output>> cached_outputs;
  std::vector<hexagon_hta_nn_input> inputs;
  std::vector<hexagon_hta_nn_output> outputs;

  for (const OperatorDef &op : net_def.op()) {
    hta_op_type op_id = op_map.GetOpId(op.type());
    inputs.resize(op.node_input().size());
    for (int i = 0; i < op.node_input().size(); ++i) {
      inputs[i].src_id = node_id(op.node_input()[i].node_id());
      inputs[i].output_idx = op.node_input()[i].output_port();
    }
    outputs.resize(op.output_shape().size());
    for (int i = 0; i < op.output_shape().size(); ++i) {
      outputs[i].rank = op.output_shape()[i].dims().size();
      for (size_t j = 0; j < outputs[i].rank; ++j) {
        outputs[i].max_sizes[j] = op.output_shape()[i].dims()[j];
      }
      if (outputs[i].rank == 0) {
        outputs[i].rank = 1;
        outputs[i].max_sizes[0] = 1;
      }
      outputs[i].max_sizes[outputs[i].rank] = 0;
      outputs[i].elementsize = GetEnumTypeSize(
          static_cast<DataType>(op.output_type()[i]));
      outputs[i].zero_offset = 0;
      outputs[i].stepsize = 0;
    }
    cached_inputs.push_back(inputs);
    cached_outputs.push_back(outputs);

    auto padding_type = static_cast<hta_padding_type>(op.padding());

    hexagon_nn_op_node op_node;
    op_node.node_id = node_id(op.node_id());
    op_node.operation = op_id;
    op_node.padding = padding_type;
    op_node.inputs = cached_inputs.back().data();
    op_node.inputsLen = inputs.size();
    op_node.outputs = cached_outputs.back().data();
    op_node.outputsLen = outputs.size();

    hexagon_hta_nn_append_node(nn_id_,
                               op_node.node_id,
                               op_node.operation,
                               op_node.padding,
                               op_node.inputs,
                               op_node.inputsLen,
                               op_node.outputs,
                               op_node.outputsLen);
  }

  // input info
  num_inputs_ = net_def.input_info_size();
  input_info_.reserve(num_inputs_);
  for (const InputOutputInfo &input_info : net_def.input_info()) {
    std::vector<index_t> input_shape(input_info.dims().begin(),
                                     input_info.dims().end());
    while (input_shape.size() < 4) {
      input_shape.insert(input_shape.begin(), 1);
    }
    input_info_.emplace_back(input_info.name(),
                             input_shape,
                             input_info.data_type(),
                             input_info.scale(),
                             input_info.zero_point(),
                             make_unique<Tensor>());
  }

  // output info
  num_outputs_ = net_def.output_info_size();
  output_info_.reserve(num_outputs_);
  for (const InputOutputInfo &output_info : net_def.output_info()) {
    std::vector<index_t> output_shape(output_info.dims().begin(),
                                      output_info.dims().end());
    while (output_shape.size() < 4) {
      output_shape.insert(output_shape.begin(), 1);
    }
    output_info_.emplace_back(output_info.name(),
                              output_shape,
                              output_info.data_type(),
                              output_info.scale(),
                              output_info.zero_point(),
                              make_unique<Tensor>());
    VLOG(1) << "OutputInfo: "
            << "\n\t shape: " << output_shape[0] << " " << output_shape[1]
            << " " << output_shape[2] << " " << output_shape[3]
            << "\n\t type: " << output_info.data_type();
  }

  int64_t t1 = NowMicros();

  MACE_CHECK(hexagon_hta_nn_prepare(nn_id_) == 0, "hexagon_nn_prepare failed");

  int64_t t2 = NowMicros();

  VLOG(1) << "Setup time: " << t1 - t0 << " " << t2 - t1;

  return true;
}

bool HexagonHTAWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_hta_nn_teardown(nn_id_) == 0;
}

void HexagonHTAWrapper::PrintLog() {
  LOG(INFO) << "Print Log";
}

void HexagonHTAWrapper::PrintGraph() {
  LOG(INFO) << "Print Graph";
}

void HexagonHTAWrapper::SetDebugLevel(int level) {
  LOG(INFO) << "Set debug level: " << level;
  MACE_CHECK(hexagon_hta_nn_set_debug_level(nn_id_, level) == 0,
             "set debug level error");
}

void HexagonHTAWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
}

void HexagonHTAWrapper::ResetPerfInfo() {
  LOG(INFO) << "Reset perf info";
}

bool HexagonHTAWrapper::ExecuteGraph(const Tensor &input_tensor,
                                     Tensor *output_tensor) {
  MACE_UNUSED(input_tensor);
  MACE_UNUSED(output_tensor);
  MACE_NOT_IMPLEMENTED;
  return false;
}

bool HexagonHTAWrapper::ExecuteGraphNew(
    const std::map<std::string, Tensor *> &input_tensors,
    std::map<std::string, Tensor *> *output_tensors) {
  VLOG(2) << "Execute graph new: " << nn_id_;
  uint32_t num_inputs = static_cast<uint32_t>(input_tensors.size());
  uint32_t num_outputs = static_cast<uint32_t>(output_tensors->size());
  MACE_CHECK(num_inputs_ == static_cast<int>(num_inputs), "Wrong inputs num");
  MACE_CHECK(num_outputs_ == static_cast<int>(num_outputs),
             "Wrong outputs num");

  std::vector<hexagon_hta_nn_tensordef> inputs(num_inputs);
  std::vector<hexagon_hta_nn_tensordef> outputs(num_outputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto input_tensor = input_tensors.at(input_info_[i].name);
    const auto &input_shape = input_tensor->shape();
    inputs[i].batches = static_cast<uint32_t>(input_shape[0]);
    inputs[i].height = static_cast<uint32_t>(input_shape[1]);
    inputs[i].width = static_cast<uint32_t>(input_shape[2]);
    inputs[i].depth = static_cast<uint32_t>(input_shape[3]);
    input_info_[i].tensor_u8->SetDtype(DT_UINT8);
    input_info_[i].tensor_u8->Resize(input_shape);

    const float *input_data = input_tensor->data<float>();
    uint8_t *input_data_u8 = input_info_[i].tensor_u8->mutable_data<uint8_t>();
    quantize_util_.QuantizeWithScaleAndZeropoint(input_data,
                                                 input_tensor->size(),
                                                 input_info_[i].scale,
                                                 input_info_[i].zero_point,
                                                 input_data_u8);

    inputs[i].data = const_cast<unsigned char *>(
        reinterpret_cast<const unsigned char *>(
            input_info_[i].tensor_u8->raw_data()));
    inputs[i].dataLen = static_cast<int>(input_info_[i].tensor_u8->raw_size());
    inputs[i].data_valid_len = static_cast<uint32_t>(
        input_info_[i].tensor_u8->raw_size());
    inputs[i].unused = 0;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_tensor = output_tensors->at(output_info_[i].name);
    output_tensor->SetDtype(output_info_[i].data_type);
    output_tensor->Resize(output_info_[i].shape);
    output_info_[i].tensor_u8->SetDtype(DT_UINT8);
    output_info_[i].tensor_u8->Resize(output_info_[i].shape);
    outputs[i].data = reinterpret_cast<unsigned char *>(
        output_info_[i].tensor_u8->raw_mutable_data());
    outputs[i].dataLen =
        static_cast<int>(output_info_[i].tensor_u8->raw_size());
  }

  int res = hexagon_hta_nn_execute_new(nn_id_,
                                       inputs.data(),
                                       num_inputs,
                                       outputs.data(),
                                       num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    std::vector<uint32_t> output_shape{
        outputs[i].batches, outputs[i].height, outputs[i].width,
        outputs[i].depth};
    MACE_CHECK(output_shape.size() == output_info_[i].shape.size(),
               output_shape.size(), " vs ", output_info_[i].shape.size(),
               "wrong output shape inferred");
    for (size_t j = 0; j < output_shape.size(); ++j) {
      MACE_CHECK(static_cast<index_t>(output_shape[j])
                     == output_info_[i].shape[j],
                 output_shape[j], " vs ", output_info_[i].shape[j],
                 "wrong output shape inferred");
    }
    auto output_tensor = output_tensors->at(output_info_[i].name);
    MACE_CHECK(static_cast<index_t>(outputs[i].data_valid_len)
                   == output_tensor->raw_size(),
               outputs[i].data_valid_len, " vs ", output_tensor->raw_size(),
               " wrong output bytes inferred.");

    const uint8_t *output_data_u8 = output_info_[i].tensor_u8->data<uint8_t>();
    float *output_data = output_tensor->mutable_data<float>();
    quantize_util_.Dequantize(output_data_u8,
                              output_info_[i].tensor_u8->size(),
                              output_info_[i].scale,
                              output_info_[i].zero_point,
                              output_data);
  }

  return res == 0;
}

}  // namespace mace
