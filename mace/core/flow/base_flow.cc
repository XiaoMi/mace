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


#include "mace/core/flow/base_flow.h"

#include <functional>

#include "mace/core/mace_tensor_impl.h"
#include "mace/core/net_def_adapter.h"
#include "mace/core/proto/net_def_helper.h"
#include "mace/utils/math.h"
#include "mace/utils/stl_util.h"
#include "mace/utils/transpose.h"

namespace mace {

BaseFlow::BaseFlow(FlowContext *flow_context)
    : net_(nullptr),
      ws_(make_unique<Workspace>(flow_context->op_delegator_registry, this)),
      is_quantized_model_(false),
      op_registry_(flow_context->op_registry),
      config_impl_(flow_context->config_impl),
      cpu_runtime_(flow_context->cpu_runtime),
      main_runtime_(flow_context->main_runtime),
      thread_pool_(flow_context->thread_pool),
      parent_engine_(flow_context->parent_engine) {}

const std::string &BaseFlow::GetName() const {
  return name_;
}

const BaseEngine *BaseFlow::GetMaceEngine() const {
  return parent_engine_;
}

MaceStatus BaseFlow::Init(const NetDef *net_def,
                          const unsigned char *model_data,
                          const int64_t model_data_size,
                          bool *model_data_unused) {
  name_ = net_def->name();
  // Mark quantized model flag
  is_quantized_model_ = NetDefHelper::IsQuantizedModel(*net_def);
  net_data_type_ = net_def->data_type();

  // Get input and output information.
  for (auto &input_info : net_def->input_info()) {
    input_info_map_[input_info.name()] = input_info;
  }
  for (auto &output_info : net_def->output_info()) {
    output_info_map_[output_info.name()] = output_info;
  }

  MACE_RETURN_IF_ERROR(InitInputTensors());
  MACE_RETURN_IF_ERROR(AllocateBufferForInputTensors());

  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  MACE_UNUSED(model_data_unused);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::Run(const std::map<std::string, MaceTensor> &inputs,
                         std::map<std::string, MaceTensor> *outputs,
                         RunMetadata *run_metadata) {
  MACE_CHECK_NOTNULL(outputs);
  TensorMap input_tensors;
  TensorMap output_tensors;

  // Create and Transpose input tensors
  for (auto &input : inputs) {
    if (input_info_map_.find(input.first) == input_info_map_.end()) {
      LOG(FATAL) << "'" << input.first
                 << "' does not belong to model's inputs: "
                 << MakeString(MapKeys(input_info_map_));
    }
    Tensor *input_tensor = ws_->GetTensor(input.first);
    MACE_RETURN_IF_ERROR(TransposeInput(input, input_tensor));
    input_tensors[input.first] = input_tensor;
  }

  // Create output tensors
  for (auto &output : *outputs) {
    if (output_info_map_.find(output.first) == output_info_map_.end()) {
      LOG(FATAL) << "'" << output.first
                 << "' does not belong to model's outputs: "
                 << MakeString(MapKeys(output_info_map_));
    }
    Tensor *output_tensor = ws_->GetTensor(output.first);
    output_tensors[output.first] = output_tensor;
  }

  // Run Model
  MACE_RETURN_IF_ERROR(Run(&input_tensors, &output_tensors, run_metadata));

  // Transpose output tensors
  for (auto &output : *outputs) {
    Tensor *output_tensor = ws_->GetTensor(output.first);
    // save output
    MACE_RETURN_IF_ERROR(TransposeOutput(*output_tensor, &output));
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::AllocateIntermediateBuffer() {
  MACE_RETURN_IF_ERROR(AllocateBufferForInputTensors());
  if (net_ != nullptr) {
    MACE_RETURN_IF_ERROR(net_->AllocateIntermediateBuffer());
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::TransposeInput(
    const std::pair<const std::string, MaceTensor> &input,
    Tensor *input_tensor) {
  std::vector<int> dst_dims;
  DataFormat data_format = DataFormat::NONE;
  MACE_RETURN_IF_ERROR(GetInputTransposeDims(
      input, input_tensor, &dst_dims, &data_format));

  // Resize the input tensor
  std::vector<index_t> output_shape = input.second.shape();
  if (!dst_dims.empty()) {
    output_shape = TransposeShape<int64_t, index_t>(input.second.shape(),
                                                    dst_dims);
  }
  MACE_RETURN_IF_ERROR(input_tensor->Resize(output_shape));

  // Transpose or copy the mace tensor's data to input tensor
  auto status = TransposeInputByDims(input.second, input_tensor, dst_dims);

  // Set the data format
  input_tensor->set_data_format(data_format);

  return status;
}

MaceStatus BaseFlow::TransposeOutput(
    const mace::Tensor &output_tensor,
    std::pair<const std::string, mace::MaceTensor> *output) {
  MACE_CHECK(output->second.data() != nullptr);

  // Get the transpose rule
  std::vector<int> dst_dims = GetOutputTransposeDims(output_tensor, output);

  // Set the output's shape
  std::vector<index_t> shape = output_tensor.shape();
  if (!dst_dims.empty()) {
    shape = TransposeShape<index_t, index_t>(output_tensor.shape(), dst_dims);
  }
  int64_t output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int64_t>());
  VLOG(1) << "output_tensor name: " << output_tensor.name();
  MACE_CHECK(output_size <= output->second.impl_->buffer_size)
    << "Output size exceeds buffer size: shape"
    << MakeString<int64_t>(shape) << " vs buffer size "
    << output->second.impl_->buffer_size;
  output->second.impl_->shape = shape;

  // Transpose output tensor
  return TransposeOutputByDims(output_tensor, &(output->second), dst_dims);
}

std::vector<int> BaseFlow::GetOutputTransposeDims(
    const mace::Tensor &output_tensor,
    std::pair<const std::string, mace::MaceTensor> *output) {
  std::vector<int> dst_dims;
  if (output_tensor.data_format() != DataFormat::NONE &&
      output->second.data_format() != DataFormat::NONE &&
      output->second.shape().size() == 4 &&
      output->second.data_format() != output_tensor.data_format()) {
    VLOG(1) << "Transpose output " << output->first << " from "
            << static_cast<int>(output_tensor.data_format()) << " to "
            << static_cast<int>(output->second.data_format());

    if (output_tensor.data_format() == DataFormat::NCHW &&
        output->second.data_format() == DataFormat::NHWC) {
      dst_dims = {0, 2, 3, 1};
    } else if (output_tensor.data_format() == DataFormat::NHWC &&
        output->second.data_format() == DataFormat::NCHW) {
      dst_dims = {0, 3, 1, 2};
    } else {
      LOG(FATAL) << "Not supported output data format: "
                 << static_cast<int>(output->second.data_format()) << " vs "
                 << static_cast<int>(output_tensor.data_format());
    }
  }
  return dst_dims;
}

MaceStatus BaseFlow::TransposeOutputByDims(const mace::Tensor &output_tensor,
                                           MaceTensor *mace_tensor,
                                           const std::vector<int> &dst_dims) {
  auto output_dt = output_tensor.dtype();
  if (!dst_dims.empty()) {
    if (output_dt == DataType::DT_INT32) {
      Tensor::MappingGuard output_guard(&output_tensor);
      auto output_data = output_tensor.data<int>();
      MACE_RETURN_IF_ERROR(ops::Transpose(
          thread_pool_, output_data, output_tensor.shape(),
          dst_dims, mace_tensor->data<int>().get()));
    } else {
      LOG(FATAL) << "MACE do not support the output data type: " << output_dt;
    }
  } else {
    if (output_dt == DataType::DT_INT32) {
      Tensor::MappingGuard output_guard(&output_tensor);
      std::memcpy(mace_tensor->data<int>().get(),
                  output_tensor.data<int>(),
                  output_tensor.size() * sizeof(int));
    } else {
      LOG(FATAL) << "MACE do not support the output data type: " << output_dt;
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

Tensor *BaseFlow::CreateInputTensor(const std::string &input_name,
                                    DataType input_dt) {
  const auto mem_type = main_runtime_->GetBaseMemoryType();
  const auto runtime_type = main_runtime_->GetRuntimeType();
  if (runtime_type == RT_OPENCL && input_dt == DT_FLOAT16) {
    // For GPU, DT_FLOAT16 is DT_HALF
    input_dt = DT_HALF;
  } else if (net_data_type_ != DT_FLOAT16 && runtime_type == RT_CPU &&
      input_dt == DT_FLOAT16) {
    // For CPU, when it is a fp16_fp16 model, use DT_FLOAT16,
    // when it is a fp16_fp32 model, use DT_FLOAT
    input_dt = DT_FLOAT;
  }

  return ws_->CreateTensor(input_name, main_runtime_, input_dt,
                           false, mem_type);
}

MaceStatus BaseFlow::GetInputTransposeDims(
    const std::pair<const std::string, MaceTensor> &input,
    const Tensor *input_tensor,
    std::vector<int> *dst_dims, DataFormat *data_format) {
  MACE_UNUSED(input_tensor);

  *dst_dims = {};
  *data_format = input.second.data_format();

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::TransposeInputByDims(const MaceTensor &mace_tensor,
                                          Tensor *input_tensor,
                                          const std::vector<int> &dst_dims) {
  DataType input_dt = input_tensor->dtype();
  if (!dst_dims.empty()) {
    if (input_dt == DataType::DT_INT32) {
      MACE_CHECK(mace_tensor.data_type() == IDT_INT32,
                 "Invalid data type.");
      Tensor::MappingGuard input_guard(input_tensor);
      auto input_data = input_tensor->mutable_data<int>();
      return ops::Transpose(thread_pool_,
                            mace_tensor.data<int>().get(),
                            mace_tensor.shape(),
                            dst_dims,
                            input_data);
    } else {
      LOG(FATAL) << "MACE do not support the input data type: " << input_dt;
    }
  } else {
    if (input_dt == DataType::DT_INT32) {
      MACE_CHECK(mace_tensor.data_type() == IDT_INT32,
                 "Invalid data type.");
      Tensor::MappingGuard input_guard(input_tensor);
      ops::CopyDataBetweenSameType(
          thread_pool_, mace_tensor.data().get(),
          input_tensor->mutable_data<int>(), input_tensor->raw_size());
    } else {
      LOG(FATAL) << "MACE do not support the input data type: " << input_dt;
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::InitInputTensors() {
  for (auto &input : input_info_map_) {
    const auto &input_name = input.first;
    const auto &input_info = input.second;
    DataType input_dt = input_info.data_type();
    Tensor *input_tensor = CreateInputTensor(input_name, input_dt);
    // Resize to possible largest shape to avoid resize during running.
    std::vector<index_t> shape(input_info.dims_size());
    for (int i = 0; i < input_info.dims_size(); ++i) {
      shape[i] = input_info.dims(i);
    }
    input_tensor->Reshape(shape);

    // Set to the default data format
    input_tensor->set_data_format(
        static_cast<DataFormat>(input_info.data_format()));
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::AllocateBufferForInputTensors() {
  for (auto &input : input_info_map_) {
    const auto &input_name = input.first;
    Tensor *input_tensor = ws_->GetTensor(input_name);

    MACE_RETURN_IF_ERROR(main_runtime_->AllocateBufferForTensor(
        input_tensor, BufRentType::RENT_SHARE));
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BaseFlow::InitOutputTensor() {
  for (auto &output : output_info_map_) {
    const auto &output_name = output.first;
    const auto &output_info = output.second;
    DataType output_dt = output_info.data_type();
    Tensor *output_tensor =
        ws_->CreateTensor(output_name, main_runtime_, output_dt);
    output_tensor->set_data_format(DataFormat::NHWC);
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace
