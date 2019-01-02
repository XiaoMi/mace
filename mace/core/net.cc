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

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>

#include "mace/core/future.h"
#include "mace/core/macros.h"
#include "mace/core/memory_optimizer.h"
#include "mace/core/net.h"
#include "mace/core/op_context.h"
#include "mace/public/mace.h"
#include "mace/utils/memory_logging.h"
#include "mace/utils/timer.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {

namespace {
struct InternalOutputInfo {
  InternalOutputInfo(const MemoryType mem_type,
                     const DataType dtype,
                     const std::vector<index_t> &shape,
                     int op_idx)
      : mem_type(mem_type), dtype(dtype), shape(shape), op_idx(op_idx) {}

  MemoryType mem_type;  // transformed memory type
  DataType dtype;
  std::vector<index_t> shape;  // tensor shape
  int op_idx;  // operation which generate the tensor
};

#ifdef MACE_ENABLE_OPENCL
std::string TransformedName(const std::string &input_name,
                            const mace::MemoryType mem_type) {
  std::stringstream ss;
  ss << input_name << "_mem_type_" << mem_type;
  return ss.str();
}

bool TransformRequiredOp(const std::string &op_type) {
  static const std::unordered_set<std::string> kNoTransformOp = {
      "Shape", "InferConv2dShape"
  };
  return kNoTransformOp.count(op_type) == 0;
}
#endif  // MACE_ENABLE_OPENCL



// TODO(lichao): Move to runtime driver class after universality done.
// fallback to gpu buffer when kernels are implemented
void FindAvailableDevicesForOp(const OpRegistryBase &op_registry,
                               const OperatorDef &op,
                               const std::unordered_map<std::string,
                                   std::vector<index_t>> &tensor_shape_info,
                               std::set<DeviceType>
                                   *available_devices) {
  auto devices = op_registry.AvailableDevices(op.type());
  available_devices->insert(devices.begin(), devices.end());
  std::string op_type = op.type();
  // For those whose shape is not 4-rank but can run on GPU
  if (op_type == "BufferTransform"
      || op_type == "LSTMCell"
      || op_type == "FullyConnected"
      || op_type == "Softmax"
      || op_type == "Squeeze") {
    return;
  } else {
    if (op.output_shape_size() != op.output_size()) {
      return;
    }
    if (op.output_shape(0).dims_size() != 4) {
      available_devices->erase(DeviceType::GPU);
    }

    if (op_type == "Split") {
      if (op.output_shape(0).dims_size() != 4
          || op.output_shape(0).dims()[3] % 4 != 0) {
        available_devices->erase(DeviceType::GPU);
      }
    } else if (op_type == "Concat") {
      if (op.output_shape(0).dims_size() != 4) {
        available_devices->erase(DeviceType::GPU);
      } else {
        if (op.input_size() != 2) {
          for (const std::string &input : op.input()) {
            if (tensor_shape_info.find(input) != tensor_shape_info.end()) {
              auto &input_shape = tensor_shape_info.at(input);
              if (input_shape[3] % 4 != 0) {
                available_devices->erase(DeviceType::GPU);
                break;
              }
            }
          }
        }
      }
    } else if (op_type == "ChannelShuffle") {
      int groups = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          op, "group", 1);
      int channels = op.output_shape(0).dims(3);
      int channels_per_group = channels / groups;
      if (groups % 4 != 0 || channels_per_group % 4 != 0) {
        available_devices->erase(DeviceType::GPU);
      }
    }
  }
}

}  // namespace

std::unique_ptr<Operation> SerialNet::CreateOperation(
    const OpRegistryBase *op_registry,
    OpConstructContext *construct_context,
    std::shared_ptr<OperatorDef> op_def,
    const std::unordered_map<std::string,
                             std::vector<index_t>> tensor_shape_info,
    DataFormat data_format_flag,
    bool is_quantize_model) {
  // Create the Operation
  DeviceType target_device_type = target_device_->device_type();
  // Get available devices
  std::set<DeviceType> available_devices;
  FindAvailableDevicesForOp(*op_registry,
                            *op_def,
                            tensor_shape_info,
                            &available_devices);
  // Find the device type to run the op.
  // If the target_device_type in available devices, use target_device_type,
  // otherwise, fallback to CPU device.
  DeviceType device_type = DeviceType::CPU;
  construct_context->set_device(cpu_device_);
  construct_context->set_operator_def(op_def);
  construct_context->set_output_mem_type(MemoryType::CPU_BUFFER);
  for (auto device : available_devices) {
    if (device == target_device_type) {
      device_type = target_device_type;
      construct_context->set_device(target_device_);
      if (target_device_->device_type() == DeviceType::GPU) {
        construct_context->set_output_mem_type(MemoryType::GPU_IMAGE);
      }
      break;
    }
  }
  op_def->set_device_type(device_type);

  // transpose output shape if run on CPU (default format is NHWC)
  if (!is_quantize_model && device_type == DeviceType::CPU &&
      op_def->output_shape_size() == op_def->output_size()) {
    for (int out_idx = 0; out_idx < op_def->output_size(); ++out_idx) {
      if (data_format_flag == NHWC &&
          op_def->output_shape(out_idx).dims_size() == 4) {
        //  NHWC -> NCHW
        std::vector<index_t> output_shape =
            TransposeShape<index_t, index_t>(
                std::vector<index_t>(
                    op_def->output_shape(out_idx).dims().begin(),
                    op_def->output_shape(out_idx).dims().end()),
                {0, 3, 1, 2});
        for (int i = 0; i < 4; ++i) {
          op_def->mutable_output_shape(out_idx)->set_dims(i, output_shape[i]);
        }
      }
    }
  }
  std::unique_ptr<Operation> op(
      op_registry->CreateOperation(construct_context, device_type));
  return std::move(op);
}

SerialNet::SerialNet(const OpRegistryBase *op_registry,
                     const NetDef *net_def,
                     Workspace *ws,
                     Device *target_device,
                     MemoryOptimizer *mem_optimizer)
    : NetBase(),
      ws_(ws),
      target_device_(target_device),
      cpu_device_(
          new CPUDevice(target_device->cpu_runtime()->num_threads(),
                        target_device->cpu_runtime()->policy(),
                        target_device->cpu_runtime()->use_gemmlowp())) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet");
  // output tensor : related information
  std::unordered_map<std::string, InternalOutputInfo> output_map;
  // used for memory optimization
  std::unordered_map<std::string, MemoryType> output_mem_map;
  std::unordered_set<std::string> transformed_set;
  // add input information
  MemoryType target_mem_type;
  // quantize model flag
  bool is_quantize_model = IsQuantizedModel(*net_def);

  DataFormat data_format_flag = NHWC;
  if (target_device_->device_type() == DeviceType::CPU) {
    target_mem_type = MemoryType::CPU_BUFFER;
    for (auto &input_info : net_def->input_info()) {
      std::vector<index_t> input_shape =
          std::vector<index_t>(input_info.dims().begin(),
                               input_info.dims().end());
      // Only could be NONE or NHWC
      auto input_data_format = static_cast<DataFormat>(
          input_info.data_format());
      if (!is_quantize_model &&
          input_data_format == NHWC &&
          input_info.dims_size() == 4) {
        // NHWC -> NCHW
        input_shape =
            TransposeShape<index_t, index_t>(input_shape, {0, 3, 1, 2});
      } else if (input_data_format == DataFormat::DF_NONE) {
        data_format_flag = DataFormat::DF_NONE;
      }
      output_map.emplace(input_info.name(), InternalOutputInfo(
          target_mem_type, DataType::DT_FLOAT, input_shape, -1));
    }
  }

#ifdef MACE_ENABLE_OPENCL
  else {  // GPU  NOLINT[readability/braces]
    target_mem_type = MemoryType::GPU_BUFFER;
    for (auto &input_info : net_def->input_info()) {
      std::vector<index_t> input_shape =
          std::vector<index_t>(input_info.dims().begin(),
                               input_info.dims().end());
      output_map.emplace(input_info.name(), InternalOutputInfo(
          target_mem_type, DataType::DT_FLOAT, input_shape, -1));
    }
  }
#endif  // MACE_ENABLE_OPENCL

  std::unordered_map<std::string, std::vector<index_t>> tensor_shape_info;
  for (auto &op : net_def->op()) {
    if (op.output_size() != op.output_shape_size()) {
      continue;
    }
    for (int i = 0; i < op.output_size(); ++i) {
      tensor_shape_info[op.output(i)] =
          std::move(std::vector<index_t>(op.output_shape(i).dims().begin(),
                                         op.output_shape(i).dims().end()));
    }
  }
  for (auto &tensor : net_def->tensors()) {
    tensor_shape_info[tensor.name()] =
        std::move(std::vector<index_t>(tensor.dims().begin(),
                                       tensor.dims().end()));
  }
  OpConstructContext construct_context(ws_);
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    std::shared_ptr<OperatorDef> op_def(new OperatorDef(net_def->op(idx)));
    // Create operation
    auto op = CreateOperation(op_registry,
                              &construct_context,
                              op_def,
                              tensor_shape_info,
                              data_format_flag,
                              is_quantize_model);
#ifdef MACE_ENABLE_OPENCL
    // Add input transform operation if necessary
    if (target_device_->device_type() == DeviceType::GPU) {
      // the outputs' memory type of the operation
      MemoryType out_mem_type = construct_context.output_mem_type();
      int input_size = op_def->input_size();
      // if op is memory-unused op, no transformation
      if (TransformRequiredOp(op_def->type())) {
        for (int i = 0; i < input_size; ++i) {
          if (output_map.count(op_def->input(i)) == 1) {
            // if op is memory-reuse op, no transformation
            if (MemoryOptimizer::IsMemoryReuseOp(op_def->type())) {
              out_mem_type = output_map.at(op_def->input(i)).mem_type;
              break;
            }
            // check whether to do transform
            MemoryType wanted_in_mem_type =
                construct_context.GetInputMemType(i);
            DataType wanted_in_dt = construct_context.GetInputDataType(i);
            if (output_map.at(op_def->input(i)).mem_type != wanted_in_mem_type
                || output_map.at(op_def->input(i)).dtype != wanted_in_dt) {
              auto t_input_name = TransformedName(op_def->input(i),
                                                  wanted_in_mem_type);
              auto &output_info = output_map.at(op_def->input(i));
              // check whether the tensor has been transformed
              if (transformed_set.count(t_input_name) == 0) {
                VLOG(1) << "Add Transform operation " << op_def->name()
                        << " to transform tensor "
                        << op_def->input(i) << "', from memory type "
                        << output_info.mem_type << " to "
                        << wanted_in_mem_type
                        << ", from Data Type " << output_info.dtype << " to "
                        << wanted_in_dt;
                std::string input_name = op_def->input(i);
                op_def->set_input(i, t_input_name);
                auto input_shape = output_info.shape;
                if (output_info.mem_type == MemoryType::CPU_BUFFER &&
                    input_shape.size() == 4) {
                  // NCHW -> NHWC
                  input_shape =
                      TransposeShape<index_t, index_t>(input_shape,
                                                       {0, 2, 3, 1});
                }
                auto transform_op_def = OpenCLUtil::CreateTransformOpDef(
                    input_name, input_shape, t_input_name,
                    wanted_in_dt, wanted_in_mem_type);
                auto transform_op = CreateOperation(
                    op_registry,
                    &construct_context,
                    transform_op_def,
                    tensor_shape_info,
                    data_format_flag);
                operators_.emplace_back(std::move(transform_op));
                transformed_set.insert(t_input_name);
                output_mem_map[t_input_name] = wanted_in_mem_type;
                // where to do graph reference count.
                mem_optimizer->UpdateTensorRef(transform_op_def.get());
              } else {
                op_def->set_input(i, t_input_name);
              }
            }
          } else {
            MACE_CHECK(ws_->GetTensor(op_def->input(i)) != nullptr
                           && ws_->GetTensor(op_def->input(i))->is_weight(),
                       "Tensor ", op_def->input(i), " of ",
                       op_def->name(), " not allocated");
          }
        }
      }
      // update the map : output_tensor -> Operation
      for (int out_idx = 0; out_idx < op_def->output_size(); ++out_idx) {
        DataType dt;
        if (op_def->output_type_size() == op_def->output_size()) {
          dt = op_def->output_type(out_idx);
        } else {
          dt = static_cast<DataType>(
              ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                  *op_def, "T", static_cast<int>(DataType::DT_FLOAT)));
        }
        output_mem_map[op_def->output(out_idx)] = out_mem_type;
        output_map.emplace(
            op_def->output(out_idx),
            InternalOutputInfo(
                out_mem_type,
                dt,
                op_def->output_shape().empty() ?
                std::vector<index_t>() :
                std::vector<index_t>(
                    op_def->output_shape(out_idx).dims().begin(),
                    op_def->output_shape(out_idx).dims().end()),
                static_cast<int>(operators_.size())));
      }
    }
#endif  // MACE_ENABLE_OPENCL
    operators_.emplace_back(std::move(op));
    // where to do graph reference count.
    mem_optimizer->UpdateTensorRef(op_def.get());
  }

#ifdef MACE_ENABLE_OPENCL
  // Transform the output tensor if necessary
  if (target_device_->device_type() == DeviceType::GPU) {
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
            target_mem_type);
        auto output_op_def =
            operators_[internal_output_info.op_idx]->operator_def();
        int output_size = output_op_def->output_size();
        for (int i = 0; i < output_size; ++i) {
          if (output_op_def->output(i) == output_info.name()) {
            output_op_def->set_output(i, t_output_name);
            // update the output : mem_type map
            output_mem_map[t_output_name] = output_mem_map[output_info.name()];
            output_mem_map[output_info.name()] = target_mem_type;
          }
        }
        auto output_data_format =
            static_cast<DataFormat>(output_info.data_format());
        auto transform_op_def = OpenCLUtil::CreateTransformOpDef(
            t_output_name,
            internal_output_info.shape,
            output_info.name(),
            output_info.data_type(),
            target_mem_type);
        auto transform_op = CreateOperation(
            op_registry,
            &construct_context,
            transform_op_def,
            tensor_shape_info,
            output_data_format);
        operators_.emplace_back(std::move(transform_op));
        // where to do graph reference count.
        mem_optimizer->UpdateTensorRef(transform_op_def.get());
      }
    }
  }
#endif  // MACE_ENABLE_OPENCL
  // Update output tensor reference
  for (auto &output_info : net_def->output_info()) {
    mem_optimizer->UpdateTensorRef(output_info.name());
  }

  // Do memory optimization
  for (auto &op : operators_) {
    VLOG(2) << "Operator " << op->debug_def().name() << "<" << op->device_type()
            << ", " << op->debug_def().type() << ">";
    mem_optimizer->Optimize(op->operator_def().get(), output_mem_map);
  }
  VLOG(1) << mem_optimizer->DebugInfo();
}

MaceStatus SerialNet::Init() {
  MACE_LATENCY_LOGGER(1, "Initializing SerialNet");
  OpInitContext init_context(ws_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    if (device_type == target_device_->device_type()) {
      init_context.set_device(target_device_);
    } else {
      init_context.set_device(cpu_device_);
    }
    // Initialize the operation
    MACE_RETURN_IF_ERROR(op->Init(&init_context));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::Run(RunMetadata *run_metadata) {
  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  OpContext context(ws_, cpu_device_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    MACE_LATENCY_LOGGER(1, "Running operator ", op->debug_def().name(),
                        "<", device_type, ", ", op->debug_def().type(),
                        ", ",
                        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                            op->debug_def(), "T", static_cast<int>(DT_FLOAT)),
                        ">");
    if (device_type == target_device_->device_type()) {
      context.set_device(target_device_);
    } else {
      context.set_device(cpu_device_);
    }

    CallStats call_stats;
    if (run_metadata == nullptr) {
      MACE_RETURN_IF_ERROR(op->Run(&context));
    } else {
      if (device_type == DeviceType::CPU) {
        call_stats.start_micros = NowMicros();
        MACE_RETURN_IF_ERROR(op->Run(&context));
        call_stats.end_micros = NowMicros();
      } else if (device_type == DeviceType::GPU) {
        StatsFuture future;
        context.set_future(&future);
        MACE_RETURN_IF_ERROR(op->Run(&context));
        future.wait_fn(&call_stats);
      }

      // Record run metadata
      std::vector<int> strides;
      int padding_type = -1;
      std::vector<int> paddings;
      std::vector<int> dilations;
      std::vector<index_t> kernels;
      std::string type = op->debug_def().type();

      if (type.compare("Conv2D") == 0 ||
          type.compare("Deconv2D") == 0 ||
          type.compare("DepthwiseConv2d") == 0 ||
          type.compare("DepthwiseDeconv2d") == 0 ||
          type.compare("Pooling") == 0) {
        strides = op->GetRepeatedArgs<int>("strides");
        padding_type = op->GetOptionalArg<int>("padding", -1);
        paddings = op->GetRepeatedArgs<int>("padding_values");
        dilations = op->GetRepeatedArgs<int>("dilations");
        if (type.compare("Pooling") == 0) {
          kernels = op->GetRepeatedArgs<index_t>("kernels");
        } else {
          kernels = op->Input(1)->shape();
        }
      } else if (type.compare("MatMul") == 0) {
        bool transpose_a = op->GetOptionalArg<bool>("transpose_a", false);
        kernels = op->Input(0)->shape();
        if (transpose_a) {
          std::swap(kernels[kernels.size()-2], kernels[kernels.size()-1]);
        }
      } else if (type.compare("FullyConnected") == 0) {
        kernels = op->Input(1)->shape();
      }

      std::vector<std::vector<int64_t>> output_shapes;
      for (auto output : op->Outputs()) {
        output_shapes.push_back(output->shape());
      }
      OperatorStats op_stats = {op->debug_def().name(), op->debug_def().type(),
                                output_shapes,
                                {strides, padding_type, paddings, dilations,
                                 kernels}, call_stats};
      run_metadata->op_stats.emplace_back(op_stats);
    }

    VLOG(3) << "Operator " << op->debug_def().name()
            << " has shape: " << MakeString(op->Output(0)->shape());

    if (EnvEnabled("MACE_LOG_TENSOR_RANGE")) {
      for (int i = 0; i < op->OutputSize(); ++i) {
        if (op->debug_def().quantize_info_size() == 0) {
          int data_type = op->GetOptionalArg("T", static_cast<int>(DT_FLOAT));
          if (data_type == static_cast<int>(DT_FLOAT)) {
            float max_v = std::numeric_limits<float>::lowest();
            float min_v = std::numeric_limits<float>::max();
            Tensor::MappingGuard guard(op->Output(i));
            auto *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
              max_v = std::max(max_v, output_data[j]);
              min_v = std::min(min_v, output_data[j]);
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                      << "@@" << min_v << "," << max_v;
          }
        } else {
          const int bin_size = 2048;
          for (int ind = 0; ind < op->debug_def().quantize_info_size(); ++ind) {
            float min_v = op->debug_def().quantize_info(ind).minval();
            float max_v = op->debug_def().quantize_info(ind).maxval();
            std::vector<int> bin_distribution(bin_size, 0);
            float bin_v = (max_v - min_v) / bin_size;
            Tensor::MappingGuard guard(op->Output(i));
            auto *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
                int index = static_cast<int>((output_data[j] - min_v) / bin_v);
                if (index < 0)
                  index = 0;
                else if (index > bin_size-1)
                  index = bin_size-1;
                bin_distribution[index]++;
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                        << "@@" << min_v << "," << max_v<< "@@"
                        << MakeString(bin_distribution);
          }
        }
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}
}  // namespace mace
