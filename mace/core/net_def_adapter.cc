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

#include "mace/core/net_def_adapter.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/ops/ops_utils.h"
#include "mace/core/ops/op_condition_context.h"
#include "mace/core/proto/net_def_helper.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/utils/math.h"

namespace mace {

namespace {
struct FilterRulerInfo {
  const int filter_idx;
  const std::vector<int> nhwc_oihw;
  const std::vector<int> nchw_oihw;
  FilterRulerInfo(int filter_index, const std::vector<int> nhwc2oihw,
                  const std::vector<int> nchw2oihw)
      : filter_idx(filter_index),
        nhwc_oihw(std::move(nhwc2oihw)), nchw_oihw(std::move(nchw2oihw)) {}
};

typedef std::unordered_map<
    std::string, std::unique_ptr<FilterRulerInfo>> FilterTransposeRuler;

FilterTransposeRuler GetFilterTransposeRuler() {
  FilterTransposeRuler filter_ruler;
  // Filter's src format is actually HWIO in tf, OIHW in others
  // For Conv2D in MACE, the dst format is OIHW
  filter_ruler.emplace("Conv2D", make_unique<FilterRulerInfo>(
      1, std::vector<int>({3, 2, 0, 1}), std::vector<int>({})));

  // Filter's src format is actually HWOI in tf, MIHW in others
  filter_ruler.emplace("Deconv2D", make_unique<FilterRulerInfo>(
      1, std::vector<int>({2, 3, 0, 1}), std::vector<int>({})));

  filter_ruler.emplace("DepthwiseConv2d", make_unique<FilterRulerInfo>(
      1, std::vector<int>({3, 2, 0, 1}), std::vector<int>({})));

  filter_ruler.emplace("DepthwiseDeconv2d", make_unique<FilterRulerInfo>(
      1, std::vector<int>({2, 3, 0, 1}), std::vector<int>({})));

  return filter_ruler;
}

DataFormat GetDefaultDataFormat(RuntimeType runtime_type,
                                bool is_quantized_model) {
  if (runtime_type == RuntimeType::RT_CPU) {
    if (is_quantized_model) {
      return DataFormat::NHWC;
    } else {
      return DataFormat::NCHW;
    }
  } else if (runtime_type == RuntimeType::RT_OPENCL) {
    return DataFormat::NHWC;
  } else {
    LOG(FATAL) << "MACE do not support the runtime " << runtime_type;
    return DataFormat::NONE;
  }
}

template<typename T>
std::string TransformedName(const std::string &input_name,
                            const std::string &tag,
                            const T value) {
  std::stringstream ss;
  ss << input_name << "_" << tag << "_" << value;
  return ss.str();
}

void BuildTransposeOpDef(
    const std::string &input_name,
    const std::string &output_name,
    const std::vector<index_t> &output_shape,
    const std::vector<int> dst_dims,
    const DataType dt,
    RuntimeType runtime_type,
    OperatorDef *op_def) {
  std::string op_name = "mace_node_" + output_name;
  op_def->set_name(op_name);
  op_def->set_type("Transpose");
  op_def->add_input(input_name);
  op_def->add_output(output_name);
  op_def->set_device_type(runtime_type);
  Argument *arg = op_def->add_arg();
  arg->set_name("dims");
  for (auto dim : dst_dims) {
    arg->add_ints(dim);
  }
  arg = op_def->add_arg();
  arg->set_name("T");
  arg->set_i(static_cast<int32_t>(dt));
  if (!output_shape.empty()) {
    OutputShape *shape = op_def->add_output_shape();
    for (auto value : output_shape) {
      shape->add_dims(value);
    }
  }
}

#ifdef MACE_ENABLE_OPENCL
RuntimeType GetRuntimeTypeByMemType(OpConditionContext *context,
                                    MemoryType mem_type) {
  if (mem_type == GPU_IMAGE || mem_type == GPU_BUFFER) {
    return RT_OPENCL;
  } else if (mem_type == CPU_BUFFER) {
    return RT_CPU;
  } else {
    return context->runtime()->GetRuntimeType();
  }
}
#endif  // MACE_ENABLE_OPENCL

}  // namespace

NetDefAdapter::NetDefAdapter(const OpRegistry *op_registry,
                             const Workspace *ws)
    : op_registry_(op_registry), ws_(ws) {}

MaceStatus NetDefAdapter::AdaptNetDef(const NetDef *net_def,
                                      Runtime *target_runtime,
                                      Runtime *cpu_runtime,
                                      NetDef *target_net_def) {
  MACE_LATENCY_LOGGER(1, "Adapting original NetDef");
  // Copy from original op_def, leave ops alone.
  target_net_def->mutable_arg()->CopyFrom(net_def->arg());
  target_net_def->mutable_tensors()->CopyFrom(net_def->tensors());
  target_net_def->mutable_input_info()->CopyFrom(net_def->input_info());
  target_net_def->mutable_output_info()->CopyFrom(net_def->output_info());

  // Quantize model flag
  bool is_quantized_model = NetDefHelper::IsQuantizedModel(*net_def);
  // tensor -> shape
  TensorShapeMap tensor_shape_map;
  // Output tensors -> information
  TensorInfoMap output_map;
  // Output tensor : related information
  std::unordered_set<std::string> transformed_set;

  for (auto &tensor : net_def->tensors()) {
    tensor_shape_map[tensor.name()] =
        std::vector<index_t>(tensor.dims().begin(), tensor.dims().end());
  }

  const auto mem_type = target_runtime->GetBaseMemoryType();
  const auto runtime_type = target_runtime->GetRuntimeType();
  DataFormat expected_data_format =
      GetDefaultDataFormat(runtime_type, is_quantized_model);
  int input_size = target_net_def->input_info_size();
  for (int i = 0; i < input_size; ++i) {
    auto input_info = target_net_def->mutable_input_info(i);
    auto input_data_format = static_cast<DataFormat>(
        input_info->data_format());
    std::vector<index_t> input_shape(input_info->dims().begin(),
                                     input_info->dims().end());
    if (input_data_format != DataFormat::NONE
        && input_data_format != expected_data_format
        && input_shape.size() == 4) {
      if (input_data_format == DataFormat::NHWC
          && expected_data_format == DataFormat::NCHW) {
        std::vector<int> dst_dims{0, 3, 1, 2};
        input_data_format = DataFormat::NCHW;
        input_shape = TransposeShape<index_t, index_t>(input_shape, dst_dims);
      } else if (input_data_format == DataFormat::NCHW
          && expected_data_format == DataFormat::NHWC) {
        std::vector<int> dst_dims{0, 2, 3, 1};
        input_data_format = DataFormat::NHWC;
        input_shape = TransposeShape<index_t, index_t>(input_shape, dst_dims);
      }
      input_info->set_data_format(static_cast<int>(input_data_format));
      int input_shape_size = input_shape.size();
      for (int j = 0; j < input_shape_size; ++j) {
        input_info->set_dims(j, input_shape[j]);
      }
    }
    tensor_shape_map.emplace(input_info->name(), input_shape);
    output_map.emplace(input_info->name(), InternalOutputInfo(
        mem_type, input_info->data_type(),
        input_data_format, input_shape, -1));
  }

  DataFormat op_output_data_format;
  MemoryType op_output_mem_type;
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    OperatorDef op_def(net_def->op(idx));
    OpConditionContext context(ws_, &tensor_shape_map);
    context.set_operator_def(&op_def);
    // Select device
    MACE_RETURN_IF_ERROR(this->AdaptDevice(&context,
                                           target_runtime,
                                           cpu_runtime,
                                           output_map,
                                           target_net_def,
                                           &op_def));

    // Adapt data type
    MACE_RETURN_IF_ERROR(this->AdaptDataType(&context, &op_def));

    auto runtime_type = static_cast<RuntimeType>(op_def.device_type());
    if (runtime_type == RuntimeType::RT_OPENCL) {
      MACE_RETURN_IF_ERROR(this->AdaptDataFormat(&context,
                                                 &op_def,
                                                 is_quantized_model,
                                                 &output_map,
                                                 &tensor_shape_map,
                                                 &transformed_set,
                                                 &op_output_data_format,
                                                 target_net_def));
      MACE_RETURN_IF_ERROR(this->AdaptMemoryType(&context,
                                                 &op_def,
                                                 &output_map,
                                                 &tensor_shape_map,
                                                 &transformed_set,
                                                 &op_output_mem_type,
                                                 target_net_def));
    } else {
      MACE_RETURN_IF_ERROR(this->AdaptMemoryType(&context,
                                                 &op_def,
                                                 &output_map,
                                                 &tensor_shape_map,
                                                 &transformed_set,
                                                 &op_output_mem_type,
                                                 target_net_def));
      MACE_RETURN_IF_ERROR(this->AdaptDataFormat(&context,
                                                 &op_def,
                                                 is_quantized_model,
                                                 &output_map,
                                                 &tensor_shape_map,
                                                 &transformed_set,
                                                 &op_output_data_format,
                                                 target_net_def));
    }
    input_size = op_def.input_size();
    for (int i = 0; i < input_size; ++i) {
      if (output_map.count(op_def.input(i)) == 1) {
        output_map.at(op_def.input(i)).consumer_op_indices.push_back(
            target_net_def->op_size());
      }
    }

    int output_size = op_def.output_size();
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
      DataType dt;
      if (op_def.output_type_size() == op_def.output_size()) {
        dt = op_def.output_type(out_idx);
      } else {
        dt = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                op_def, "T", static_cast<int>(DataType::DT_FLOAT)));
      }
      auto output_shape = op_def.output_shape().empty() ?
                          std::vector<index_t>() :
                          std::vector<index_t>(
                              op_def.output_shape(out_idx).dims().begin(),
                              op_def.output_shape(out_idx).dims().end());
      output_map.emplace(
          op_def.output(out_idx),
          InternalOutputInfo(
              op_output_mem_type,
              dt,
              op_output_data_format,
              output_shape,
              target_net_def->op_size()));
      tensor_shape_map.emplace(op_def.output(out_idx), output_shape);
    }
    // Add op to target net
    target_net_def->add_op()->CopyFrom(op_def);
  }

  // For outputs' convert
  auto target_runtime_type = target_runtime->GetRuntimeType();
  if (target_runtime_type != RuntimeType::RT_CPU) {
    // Add buffer transform for GPU if necessary
    MemoryType target_mem_type = target_runtime->GetBaseMemoryType();
    for (auto &output_info : net_def->output_info()) {
      auto output_data_type = output_info.data_type();
      if (output_data_type == DT_FLOAT16) {
        output_data_type = DT_HALF;
      }
      auto &internal_output_info = output_map.at(output_info.name());
      auto internal_omt = internal_output_info.mem_type;
      if ((internal_omt != target_mem_type &&
          internal_omt != MemoryType::CPU_BUFFER) ||
          internal_output_info.dtype != output_info.data_type()) {
        VLOG(1) << "Add Transform operation to transform output tensor '"
                << output_info.name() << "', from memory type "
                << internal_output_info.mem_type
                << " to " << target_mem_type
                << ", from Data Type " << internal_output_info.dtype
                << " to " << output_info.data_type();
        std::string t_output_name = TransformedName(output_info.name(),
                                                    "mem_type",
                                                    target_mem_type);
        auto output_op_def = target_net_def->mutable_op(
            internal_output_info.op_idx);
        int output_size = output_op_def->output_size();
        for (int i = 0; i < output_size; ++i) {
          if (output_op_def->output(i) == output_info.name()) {
            output_op_def->set_output(i, t_output_name);
          }
        }
        for (int idx : internal_output_info.consumer_op_indices) {
          auto consumer_op_def = target_net_def->mutable_op(idx);
          int input_size = consumer_op_def->input_size();
          for (int i = 0; i < input_size; ++i) {
            if (consumer_op_def->input(i) == output_info.name()) {
              consumer_op_def->set_input(i, t_output_name);
            }
          }
        }
        auto transformed_op_def = target_net_def->add_op();
        OpsUtils::BuildTransformOpDef(
            t_output_name,
            internal_output_info.shape,
            output_info.name(),
            target_runtime_type,
            output_data_type,
            BufferContentType::IN_OUT_CHANNEL,
            target_mem_type,
            internal_output_info.data_format,
            transformed_op_def);
        // Set data format arg
        SetProtoArg<int>(
            transformed_op_def,
            "data_format",
            static_cast<int>(internal_output_info.data_format));
        // Set output memory type argument
        SetProtoArg<int>(transformed_op_def,
                         OutputMemoryTypeTagName(),
                         target_mem_type);
      }
    }
  }

  VLOG(3) << DebugString(target_net_def);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus NetDefAdapter::AdaptDevice(OpConditionContext *context,
                                      Runtime *target_runtime,
                                      Runtime *cpu_runtime,
                                      const TensorInfoMap &output_map,
                                      const NetDef *net_def,
                                      OperatorDef *op_def) {
  VLOG(3) << "Adapt device for op " << op_def->name();
  RuntimeType target_runtime_type = target_runtime->GetRuntimeType();
  VLOG(3) << "target_runtime_type: " << static_cast<int>(target_runtime_type);
  RuntimeType runtime_type = RuntimeType::RT_CPU;
  context->set_runtime(target_runtime);
  if (target_runtime_type != RuntimeType::RT_CPU) {
    std::vector<RuntimeType> producer_runtimes;
    for (auto input : op_def->input()) {
      if (output_map.count(input) == 1) {
        if (output_map.at(input).op_idx < 0) {
          producer_runtimes.push_back(target_runtime_type);
        } else {
          producer_runtimes.push_back(
              static_cast<RuntimeType>(
                  net_def->op(output_map.at(input).op_idx).device_type()));
        }
      }
    }
    // Get available runtimes
    auto available_runtimes =
        op_registry_->AvailableRuntimes(op_def->type(), context);
    runtime_type = net_optimizer_.SelectBestRuntime(op_def,
                                                    target_runtime_type,
                                                    available_runtimes,
                                                    producer_runtimes);
    if (runtime_type != target_runtime_type) {
      context->set_runtime(cpu_runtime);
      SetProtoArg<int>(op_def, IsFallbackTagName(), 1);
      LOG(INFO) << "Op " << op_def->name() << "(" << op_def->type() << ")"
                << " fall back to CPU";
    }
  }
  op_def->set_device_type(runtime_type);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus NetDefAdapter::AdaptDataType(OpConditionContext *context,
                                        OperatorDef *op_def) {
  MACE_UNUSED(context);
  // Where to add logic to support mixing precision
  // Adjust data type of op ran on CPU
  DataType dtype = static_cast<DataType>(
      ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *op_def, "T", static_cast<int>(DT_FLOAT)));
  auto runtime_type = static_cast<RuntimeType>(op_def->device_type());
  if (runtime_type == RuntimeType::RT_CPU && dtype == DT_HALF) {
    SetProtoArg<int>(op_def, "T", static_cast<int>(DataType::DT_FLOAT));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus NetDefAdapter::AdaptDataFormat(
    OpConditionContext *context,
    OperatorDef *op_def,
    bool is_quantized_model,
    TensorInfoMap *output_map,
    TensorShapeMap *tensor_shape_map,
    std::unordered_set<std::string> *transformed_set,
    DataFormat *op_output_df,
    NetDef *target_net_def) {
  VLOG(3) << "Adapt data format for op " << op_def->name();
  DataFormat op_data_format =
      static_cast<DataFormat>(ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *op_def, "data_format",
          static_cast<int>(DataFormat::NONE)));
  auto runtime_type = static_cast<RuntimeType>(op_def->device_type());
  // Adjust the data format of operation
  if (op_data_format == DataFormat::AUTO) {
    op_data_format = GetDefaultDataFormat(runtime_type, is_quantized_model);
    SetProtoArg<int>(op_def, "data_format", static_cast<int>(op_data_format));
    if (op_data_format == DataFormat::NCHW) {
      int output_shape_size = op_def->output_shape_size();
      for (int i = 0; i < output_shape_size; ++i) {
        auto output_shape = op_def->mutable_output_shape(i);
        if (output_shape->dims_size() == 4) {
          // transpose output shape format from NHWC to NCHW
          int64_t height = output_shape->dims(1);
          int64_t width = output_shape->dims(2);
          output_shape->set_dims(1, output_shape->dims(3));
          output_shape->set_dims(2, height);
          output_shape->set_dims(3, width);
        }
      }
    }
  }
  *op_output_df = op_data_format;

  // The output memory type of transpose op is based
  // on the consumer op's runtime
  MemoryType target_mem_type = MemoryType::CPU_BUFFER;
  if (runtime_type == RuntimeType::RT_OPENCL) {
    target_mem_type = MemoryType::GPU_BUFFER;
  }
  auto inputs_data_format = op_registry_->InputsDataFormat(op_def->type(),
                                                           context);
  DataFormat src_df, dst_df;
  int input_size = op_def->input_size();
  for (int i = 0; i < input_size; ++i) {
    if (output_map->count(op_def->input(i)) == 0) {
      // Check this input is const tensor(filter)
      MACE_CHECK(ws_->GetTensor(op_def->input(i)) != nullptr
                     && ws_->GetTensor(op_def->input(i))->is_weight(),
                 "Tensor ", op_def->input(i), " of ",
                 op_def->name(), " is not allocated by Workspace ahead");
      continue;
    }

    src_df = output_map->at(op_def->input(i)).data_format;
    dst_df = inputs_data_format[i];

    const std::vector<int> dst_dims =
        GetDstDimsFromTransposeRuler(output_map, op_def, i, src_df, dst_df);
    if (dst_dims.size() > 0) {
      AddTranposeOpForDataFormat(output_map, tensor_shape_map, transformed_set,
                                 target_net_def, runtime_type, target_mem_type,
                                 op_def, i, dst_df, dst_dims);
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus NetDefAdapter::AdaptMemoryType(
    OpConditionContext *context,
    OperatorDef *op_def,
    NetDefAdapter::TensorInfoMap *output_map,
    TensorShapeMap *tensor_shape_map,
    std::unordered_set<std::string> *transformed_set,
    MemoryType *op_output_mem_types,
    NetDef *target_net_def) {
  VLOG(3) << "Adapt memory type for op " << op_def->name();
  // Get expected output memory type
  // (only support one kind of memory type for multiple outputs)
  op_registry_->GetInOutMemoryTypes(op_def->type(), context);

#ifdef MACE_ENABLE_OPENCL
  int input_size = op_def->input_size();
  for (int i = 0; i < input_size; ++i) {
    if (output_map->count(op_def->input(i)) == 0) {
      MACE_CHECK(ws_->GetTensor(op_def->input(i)) != nullptr
                     && ws_->GetTensor(op_def->input(i))->is_weight(),
                 "Tensor ", op_def->input(i), " of ",
                 op_def->name(), " not allocated");
      continue;
    }
    auto &input_info = output_map->at(op_def->input(i));
    // Check whether to do transform
    MemoryType src_mem_type = input_info.mem_type;
    MemoryType dst_mem_type = context->GetInputMemType(i);
    auto wanted_input_dtype = context->GetInputDataType(i);
    if (src_mem_type != dst_mem_type ||
        (input_info.dtype != wanted_input_dtype &&
            (src_mem_type != MemoryType::CPU_BUFFER
                || dst_mem_type != MemoryType::CPU_BUFFER))) {
      // GPU_IMAGE => CPU_BUFFER change to GPU_IMAGE =>GPU_BUFFER
      if (src_mem_type == GPU_IMAGE) {
        if (dst_mem_type == CPU_BUFFER) {
          dst_mem_type = GPU_BUFFER;
          context->SetInputInfo(i, dst_mem_type, wanted_input_dtype);
        }
      }
      auto runtime_type = GetRuntimeTypeByMemType(context, dst_mem_type);

      auto transformed_name = TransformedName(op_def->input(i),
                                              "mem_type",
                                              dst_mem_type);
      // Check whether the tensor has been transformed
      if (transformed_set->count(transformed_name) == 0) {
        VLOG(1) << "Add Transform operation for " << op_def->name()
                << " to transform its tensor "
                << op_def->input(i) << "', from memory type "
                << src_mem_type << " to " << dst_mem_type;
        OperatorDef *transformed_op_def = target_net_def->add_op();
        OpsUtils::BuildTransformOpDef(
            op_def->input(i),
            input_info.shape,
            transformed_name,
            runtime_type,
            wanted_input_dtype,
            context->GetInputBufferContentType(i),
            dst_mem_type,
            input_info.data_format,
            transformed_op_def);
        // Set data format arg
        SetProtoArg<int>(transformed_op_def,
                         "data_format",
                         static_cast<int>(input_info.data_format));
        // Set output memory type argument
        SetProtoArg<int>(transformed_op_def,
                         OutputMemoryTypeTagName(),
                         dst_mem_type);

        // Update tensor consumer information
        output_map->at(op_def->input(i)).consumer_op_indices.push_back(
            target_net_def->op_size() - 1);

        // Update output information map
        output_map->emplace(
            transformed_name,
            InternalOutputInfo(
                dst_mem_type,
                context->GetInputDataType(i),
                input_info.data_format,
                input_info.shape,
                target_net_def->op_size() - 1));
        // Update tensor shape map
        tensor_shape_map->emplace(transformed_name, input_info.shape);
        // Record transformed tensors
        transformed_set->insert(transformed_name);
      }
      // Update original op_def's input
      op_def->set_input(i, transformed_name);
    }
  }
#else
  MACE_UNUSED(output_map);
  MACE_UNUSED(tensor_shape_map);
  MACE_UNUSED(transformed_set);
  MACE_UNUSED(target_net_def);
#endif  // MACE_ENABLE_OPENCL

  *op_output_mem_types = context->output_mem_type();
  SetProtoArg<int>(op_def,
                   OutputMemoryTypeTagName(),
                   context->output_mem_type());
  return MaceStatus::MACE_SUCCESS;
}

std::vector<int> NetDefAdapter::GetDstDimsFromTransposeRuler(
    TensorInfoMap *output_map, const OperatorDef *op_def, const int input_idx,
    const DataFormat src_df, const DataFormat dst_df) {
  std::vector<int> dst_dims;
  if (src_df == DataFormat::NONE || dst_df == DataFormat::NONE
      || output_map->at(op_def->input(input_idx)).shape.size() != 4) {
    return dst_dims;
  }

  if (src_df != dst_df) {  // For other operators
    bool transposable = false;
    if (src_df == DataFormat::NCHW && dst_df == DataFormat::NHWC) {
      dst_dims = {0, 2, 3, 1};
      transposable = true;
    } else if (src_df == DataFormat::NHWC && dst_df == DataFormat::NCHW) {
      dst_dims = {0, 3, 1, 2};
      transposable = true;
    } else if (dst_df == DataFormat::OIHW) {
      static const auto filter_transpose_ruler = GetFilterTransposeRuler();
      auto &op_type = op_def->type();
      MACE_CHECK((filter_transpose_ruler.count(op_type) > 0) &&
          filter_transpose_ruler.at(op_type)->filter_idx == input_idx);
      if (src_df == DataFormat::NCHW) {
        dst_dims = filter_transpose_ruler.at(op_type)->nchw_oihw;
        transposable = true;
      } else if (src_df == DataFormat::NHWC) {
        dst_dims = filter_transpose_ruler.at(op_type)->nhwc_oihw;
        transposable = true;
      }
    }
    if (!transposable) {
      LOG(FATAL) << "Encounter unsupported data format transpose from "
                 << static_cast<int>(src_df) << " to "
                 << static_cast<int>(dst_df);
    }
  }

  return dst_dims;
}

MaceStatus NetDefAdapter::AddTranposeOpForDataFormat(
    TensorInfoMap *output_map, TensorShapeMap *tensor_shape_map,
    std::unordered_set<std::string> *transformed_set, NetDef *target_net_def,
    RuntimeType runtime_type, MemoryType target_mem_type, OperatorDef *op_def,
    const int i, const DataFormat dst_df, const std::vector<int> &dst_dims) {
  std::string transformed_name = TransformedName(
      op_def->input(i), "data_format", MakeString(dst_dims));
  if (transformed_set->count(transformed_name) == 0) {
    auto &input_info = output_map->at(op_def->input(i));
    auto output_shape = input_info.shape.empty() ?
                        std::vector<index_t>() :
                        TransposeShape<index_t, index_t>(input_info.shape,
                                                         dst_dims);
    OperatorDef *transpose_op_def = target_net_def->add_op();
    BuildTransposeOpDef(op_def->input(i), transformed_name, output_shape,
                        dst_dims, input_info.dtype, runtime_type,
                        transpose_op_def);
    // Set data format arg
    SetProtoArg<int>(transpose_op_def, "data_format", static_cast<int>(dst_df));
    // Set output memory type argument
    SetProtoArg<int>(transpose_op_def,
                     OutputMemoryTypeTagName(), target_mem_type);
    // Update tensor consumer information
    output_map->at(op_def->input(i)).consumer_op_indices.push_back(
        target_net_def->op_size() - 1);

    // Update output information map
    output_map->emplace(transformed_name, InternalOutputInfo(
        target_mem_type, input_info.dtype, dst_df, output_shape,
        target_net_def->op_size() - 1));
    // Update tensor shape map
    tensor_shape_map->emplace(transformed_name, output_shape);
    // Record transformed tensors
    transformed_set->insert(transformed_name);
  }
  // Update original op_def's input
  op_def->set_input(i, transformed_name);
  return MaceStatus::MACE_SUCCESS;
}

std::string NetDefAdapter::DebugString(const NetDef *net_def) {
  std::stringstream sstream;
  auto RuntimeTypeToStrFunc = [](RuntimeType runtime_type) -> std::string {
    if (runtime_type == RuntimeType::RT_CPU) {
      return "CPU";
    } else if (runtime_type == RuntimeType::RT_OPENCL) {
      return "GPU";
    } else {
      return "Unknown";
    }
  };
  auto MemoryTypeToStrFunc = [](MemoryType type) -> std::string {
    if (type == MemoryType::CPU_BUFFER) {
      return "CPU_BUFFER";
    } else if (type == MemoryType::GPU_BUFFER) {
      return "GPU_BUFFER";
    } else if (type == MemoryType::GPU_IMAGE) {
      return "GPU_IMAGE";
    } else {
      return "Unknown";
    }
  };
  auto DataFormatToStrFunc = [](DataFormat type) -> std::string {
    if (type == DataFormat::NHWC) {
      return "NHWC";
    } else if (type == DataFormat::NCHW) {
      return "NCHW";
    } else if (type == DataFormat::NONE) {
      return "NONE";
    } else if (type == DataFormat::AUTO) {
      return "AUTO";
    } else if (type == DataFormat::OIHW) {
      return "OIHW";
    } else {
      return "Unknown";
    }
  };
  for (auto &op : net_def->op()) {
    std::string runtime_type = RuntimeTypeToStrFunc(
        static_cast<RuntimeType>(op.device_type()));
    auto dt = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op, "T", static_cast<int>(DT_FLOAT));
    std::string data_type = DataTypeToString(static_cast<DataType>(dt));
    std::string mem_type = MemoryTypeToStrFunc(
        static_cast<MemoryType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                op, OutputMemoryTypeTagName(),
                static_cast<int>(MemoryType::CPU_BUFFER))));
    std::string data_format = DataFormatToStrFunc(
        static_cast<DataFormat>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                op, "data_format", static_cast<int>(DataFormat::NONE))));

    sstream << std::endl;
    sstream << "{" << std::endl;
    sstream << "  name: " << op.name() << std::endl;
    sstream << "  type: " << op.type() << std::endl;
    sstream << "  runtime: " << runtime_type << std::endl;
    sstream << "  data type: " << data_type << std::endl;
    sstream << "  data format: " << data_format << std::endl;
    sstream << "  memory type: " << mem_type << std::endl;
    sstream << "  inputs: [";
    for (auto input : op.input()) {
      sstream << input << ", ";
    }
    sstream << "]" << std::endl;
    sstream << "  outputs: [";
    for (auto output : op.output()) {
      sstream << output << ", ";
    }
    sstream << "]" << std::endl;
    sstream << "  output shapes: [";
    for (auto output_shape : op.output_shape()) {
      sstream << "(";
      for (auto dim : output_shape.dims())
        sstream << dim << ",";
      sstream << ") ";
    }
    sstream << "]" << std::endl;
    sstream << "}";
  }
  return sstream.str();
}

}  // namespace mace
