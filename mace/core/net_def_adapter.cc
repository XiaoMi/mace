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
#include <vector>

#include "mace/core/operator.h"
#include "mace/utils/math.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL
namespace mace {

namespace {
DataFormat GetDefaultDataFormat(DeviceType device_type,
                                bool is_quantized_model) {
  if (device_type == CPU) {
    if (is_quantized_model) {
      return DataFormat::NHWC;
    } else {
      return DataFormat::NCHW;
    }
  } else if (device_type == GPU) {
    return DataFormat::NHWC;
  } else {
    LOG(FATAL) << "MACE do not support the device " << device_type;
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

#ifdef MACE_ENABLE_OPENCL
bool TransformRequiredOp(const std::string &op_type) {
  static const std::unordered_set<std::string> kNoTransformOp = {
      "Shape", "InferConv2dShape"
  };
  return kNoTransformOp.count(op_type) == 0;
}
#endif  // MACE_ENABLE_OPENCL

void BuildTransposeOpDef(
    const std::string &input_name,
    const std::string &output_name,
    const std::vector<index_t> &output_shape,
    const std::vector<int> dst_dims,
    const DataType dt,
    DeviceType device_type,
    OperatorDef *op_def) {
  std::string op_name = "mace_node_" + output_name;
  op_def->set_name(op_name);
  op_def->set_type("Transpose");
  op_def->add_input(input_name);
  op_def->add_output(output_name);
  op_def->set_device_type(device_type);
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

}  // namespace

NetDefAdapter::NetDefAdapter(const OpRegistryBase *op_registry,
                             const Workspace *ws)
    : op_registry_(op_registry), ws_(ws) {}

MaceStatus NetDefAdapter::AdaptNetDef(
    const NetDef *net_def,
    Device *target_device,
    NetDef *target_net_def) {
  MACE_LATENCY_LOGGER(1, "Adapting original NetDef");
  // Copy from original op_def, leave ops alone.
  target_net_def->mutable_arg()->CopyFrom(net_def->arg());
  target_net_def->mutable_tensors()->CopyFrom(net_def->tensors());
  target_net_def->mutable_input_info()->CopyFrom(net_def->input_info());
  target_net_def->mutable_output_info()->CopyFrom(net_def->output_info());

  std::unique_ptr<CPUDevice> cpu_device = make_unique<CPUDevice>(
      target_device->cpu_runtime()->num_threads(),
      target_device->cpu_runtime()->policy(),
      &(target_device->cpu_runtime()->thread_pool()));

  // quantize model flag
  bool is_quantized_model = IsQuantizedModel(*net_def);
  // tensor -> shape
  TensorShapeMap tensor_shape_map;
  // Output tensors -> information
  TensorInfoMap output_map;
  // output tensor : related information
  std::unordered_set<std::string> transformed_set;

  for (auto &tensor : net_def->tensors()) {
    tensor_shape_map[tensor.name()] =
        std::vector<index_t>(tensor.dims().begin(), tensor.dims().end());
  }

  MemoryType mem_type = MemoryType::CPU_BUFFER;
  if (target_device->device_type() == DeviceType::CPU) {
    mem_type = MemoryType::CPU_BUFFER;
  } else if (target_device->device_type() == DeviceType::GPU) {
    mem_type = MemoryType::GPU_BUFFER;
  } else {
    LOG(FATAL) << "MACE do not support the device type: "
               << target_device->device_type();
  }

  DataFormat expected_data_format = GetDefaultDataFormat(
      target_device->device_type(), is_quantized_model);
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

  OpConditionContext context(ws_, &tensor_shape_map);
  DataFormat op_output_data_format;
  MemoryType op_output_mem_type;
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    OperatorDef op_def(net_def->op(idx));
    context.set_operator_def(&op_def);
    // Select device
    MACE_RETURN_IF_ERROR(this->AdaptDevice(&context,
                                           target_device,
                                           cpu_device.get(),
                                           output_map,
                                           target_net_def,
                                           &op_def));

    // Adapt data type
    MACE_RETURN_IF_ERROR(this->AdaptDataType(&context,
                                             &op_def));

    if (op_def.device_type() == DeviceType::GPU) {
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

#ifdef MACE_ENABLE_OPENCL
  if (target_device->device_type() == DeviceType::GPU) {
    // Add buffer transform for GPU if necessary
    MemoryType target_mem_type = MemoryType::GPU_BUFFER;
    for (auto &output_info : net_def->output_info()) {
      auto &internal_output_info = output_map.at(output_info.name());
      if ((internal_output_info.mem_type != target_mem_type &&
          internal_output_info.mem_type != MemoryType::CPU_BUFFER) ||
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
        auto transformed_op_def = target_net_def->add_op();
        OpenCLUtil::BuildTransformOpDef(
            t_output_name,
            internal_output_info.shape,
            output_info.name(),
            output_info.data_type(),
            OpenCLBufferType::IN_OUT_CHANNEL,
            target_mem_type,
            internal_output_info.data_format,
            transformed_op_def);
        // set data format arg
        SetProtoArg<int>(
            transformed_op_def,
            "data_format",
            static_cast<int>(internal_output_info.data_format));
        // set output memory type argument
        SetProtoArg<int>(transformed_op_def,
                         OutputMemoryTypeTagName(),
                         target_mem_type);
      }
    }
  }
#endif  // MACE_ENABLE_OPENCL

  VLOG(3) << DebugString(target_net_def);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus NetDefAdapter::AdaptDevice(OpConditionContext *context,
                                      Device *target_device,
                                      Device *cpu_device,
                                      const TensorInfoMap &output_map,
                                      const NetDef *net_def,
                                      OperatorDef *op_def) {
  VLOG(3) << "Adapt device for op " << op_def->name();
  DeviceType target_device_type = target_device->device_type();
  DeviceType device_type = DeviceType::CPU;
  context->set_device(cpu_device);
  if (target_device_type != DeviceType::CPU) {
    std::vector<DeviceType> producer_devices;
    for (auto input : op_def->input()) {
      if (output_map.count(input) == 1) {
        if (output_map.at(input).op_idx < 0) {
          producer_devices.push_back(target_device_type);
        } else {
          producer_devices.push_back(
              static_cast<DeviceType>(
                  net_def->op(output_map.at(input).op_idx).device_type()));
        }
      }
    }
    // Get available devices
    auto available_devices =
        op_registry_->AvailableDevices(op_def->type(), context);
    device_type = net_optimizer_.SelectBestDevice(op_def,
                                                  target_device_type,
                                                  available_devices,
                                                  producer_devices);
    if (device_type == target_device_type) {
      context->set_device(target_device);
    } else {
      LOG(INFO) << "Op " << op_def->name() << " fall back to CPU";
    }
  }
  op_def->set_device_type(device_type);
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
  if (op_def->device_type() == DeviceType::CPU && dtype == DT_HALF) {
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
  // adjust the data format of operation
  if (op_data_format == DataFormat::AUTO) {
    op_data_format = GetDefaultDataFormat(
        static_cast<DeviceType>(op_def->device_type()), is_quantized_model);
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

  // the output memory type of transpose op is based on the consumer op's device
  MemoryType target_mem_type = MemoryType::CPU_BUFFER;
  if (op_def->device_type() == DeviceType::GPU) {
    target_mem_type = MemoryType::GPU_BUFFER;
  }
  auto inputs_data_format = op_registry_->InputsDataFormat(op_def->type(),
      context);
  DataFormat src_df, dst_df;
  int input_size = op_def->input_size();
  for (int i = 0; i < input_size; ++i) {
    if (output_map->count(op_def->input(i)) == 0) {
      // check this input is const tensor(filter)
      MACE_CHECK(ws_->GetTensor(op_def->input(i)) != nullptr
                     && ws_->GetTensor(op_def->input(i))->is_weight(),
                 "Tensor ", op_def->input(i), " of ",
                 op_def->name(), " is not allocated by Workspace ahead");
      continue;
    }
    src_df = output_map->at(op_def->input(i)).data_format;
    dst_df = inputs_data_format[i];
    if (src_df == DataFormat::NONE
        || dst_df == DataFormat::NONE
        || output_map->at(op_def->input(i)).shape.size() != 4) {
      continue;
    }
    if (src_df != dst_df) {
      std::string transformed_name = TransformedName(op_def->input(i),
          "data_format", static_cast<int>(dst_df));
      if (transformed_set->count(transformed_name) == 0) {
        VLOG(1) << "Add Transpose operation " << op_def->name()
                << " to transpose tensor "
                << op_def->input(i) << "', from data format "
                << static_cast<int>(src_df) << " to "
                << static_cast<int>(dst_df);
        // Only support transpose between NHWC and NCHW for now.
        std::vector<int> dst_dims;
        if (src_df == DataFormat::NCHW && dst_df == DataFormat::NHWC) {
          dst_dims = {0, 2, 3, 1};
        } else if (src_df == DataFormat::NHWC && dst_df == DataFormat::NCHW) {
          dst_dims = {0, 3, 1, 2};
        } else {
          LOG(FATAL) << "Encounter unsupported data format transpose from "
                     << static_cast<int>(src_df) << " to "
                     << static_cast<int>(dst_df);
        }
        auto &input_info = output_map->at(op_def->input(i));
        auto output_shape = input_info.shape.empty() ?
                            std::vector<index_t>() :
                            TransposeShape<index_t, index_t>(input_info.shape,
                                                             dst_dims);
        OperatorDef *transpose_op_def = target_net_def->add_op();
        BuildTransposeOpDef(
            op_def->input(i),
            transformed_name,
            output_shape,
            dst_dims,
            input_info.dtype,
            DeviceType::CPU,
            transpose_op_def);
        // set data format arg
        SetProtoArg<int>(transpose_op_def,
                         "data_format",
                         static_cast<int>(dst_df));
        // set output memory type argument
        SetProtoArg<int>(transpose_op_def,
                         OutputMemoryTypeTagName(),
                         target_mem_type);

        // update output information map
        output_map->emplace(
            transformed_name,
            InternalOutputInfo(
                target_mem_type,
                input_info.dtype,
                dst_df,
                output_shape,
                target_net_def->op_size() - 1));
        // update tensor shape map
        tensor_shape_map->emplace(transformed_name, output_shape);
        // record transformed tensors
        transformed_set->insert(transformed_name);
      }
      // update original op_def's input
      op_def->set_input(i, transformed_name);
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
  // if op is memory-unused op, no transformation
  if (TransformRequiredOp(op_def->type())) {
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
      // check whether to do transform
      MemoryType src_mem_type = input_info.mem_type;
      MemoryType dst_mem_type = context->GetInputMemType(i);
      auto wanted_input_dtype = context->GetInputDataType(i);
      if (src_mem_type != dst_mem_type ||
          (input_info.dtype != wanted_input_dtype &&
              (src_mem_type != MemoryType::CPU_BUFFER
                  || dst_mem_type != MemoryType::CPU_BUFFER))) {
        auto transformed_name = TransformedName(op_def->input(i),
                                                "mem_type",
                                                dst_mem_type);
        // check whether the tensor has been transformed
        if (transformed_set->count(transformed_name) == 0) {
          VLOG(1) << "Add Transform operation " << op_def->name()
                  << " to transform tensor "
                  << op_def->input(i) << "', from memory type "
                  << input_info.mem_type << " to "
                  << dst_mem_type;
          OperatorDef *transformed_op_def = target_net_def->add_op();
          OpenCLUtil::BuildTransformOpDef(
              op_def->input(i),
              input_info.shape,
              transformed_name,
              wanted_input_dtype,
              context->GetInputOpenCLBufferType(i),
              dst_mem_type,
              input_info.data_format,
              transformed_op_def);
          // set data format arg
          SetProtoArg<int>(transformed_op_def,
                           "data_format",
                           static_cast<int>(input_info.data_format));
          // set output memory type argument
          SetProtoArg<int>(transformed_op_def,
                           OutputMemoryTypeTagName(),
                           dst_mem_type);

          // update output information map
          output_map->emplace(
              transformed_name,
              InternalOutputInfo(
                  dst_mem_type,
                  context->GetInputDataType(i),
                  input_info.data_format,
                  input_info.shape,
                  target_net_def->op_size() - 1));
          // update tensor shape map
          tensor_shape_map->emplace(transformed_name, input_info.shape);
          // record transformed tensors
          transformed_set->insert(transformed_name);
        }
        // update original op_def's input
        op_def->set_input(i, transformed_name);
      }
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

std::string NetDefAdapter::DebugString(const NetDef *net_def) {
  std::stringstream sstream;
  auto DeviceTypeToStrFunc = [](DeviceType device_type) -> std::string {
    if (device_type == DeviceType::CPU) {
      return "CPU";
    } else if (device_type == DeviceType::GPU) {
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
    std::string device_type = DeviceTypeToStrFunc(
        static_cast<DeviceType>(op.device_type()));
    std::string data_type = DataTypeToString(static_cast<DataType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "T", static_cast<int>(DT_FLOAT))));
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
    sstream << "  name: "        << op.name() << std::endl;
    sstream << "  type: "        << op.type() << std::endl;
    sstream << "  device: "      << device_type << std::endl;
    sstream << "  data type: "   << data_type << std::endl;
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
