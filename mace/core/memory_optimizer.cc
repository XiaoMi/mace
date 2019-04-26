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

#include "mace/core/memory_optimizer.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "mace/core/arg_helper.h"
#include "mace/utils/macros.h"
#include "mace/utils/logging.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {

bool MemoryOptimizer::IsMemoryReuseOp(const std::string &op_type) {
  static const std::unordered_set<std::string> kReuseOp = {
      "Reshape", "Identity", "Squeeze", "ExpandDims"
  };
  return kReuseOp.count(op_type) == 1;
}

void MemoryOptimizer::UpdateTensorRef(const std::string &tensor_name) {
  if (tensor_ref_count_.count(tensor_name) == 0) {
    tensor_ref_count_.emplace(tensor_name, 1);
  } else {
    tensor_ref_count_[tensor_name] += 1;
  }
}

void MemoryOptimizer::UpdateTensorRef(const mace::OperatorDef *op_def) {
  int input_size = op_def->input_size();
  for (int i = 0; i < input_size; ++i) {
    if (tensor_ref_count_.count(op_def->input(i)) == 1) {
      tensor_ref_count_[op_def->input(i)] += 1;
    }
  }
  int output_size = op_def->output_size();
  for (int i = 0; i < output_size; ++i) {
    if (tensor_ref_count_.count(op_def->output(i)) == 0) {
      tensor_ref_count_.emplace(op_def->output(i), 0);
    }
  }
}

MemoryBlock MemoryOptimizer::CreateMemoryBlock(
    const OperatorDef *op_def,
    int output_idx,
    DataType dt,
    MemoryType mem_type) {
  auto shape = std::vector<int64_t>(
      op_def->output_shape(output_idx).dims().begin(),
      op_def->output_shape(output_idx).dims().end());
  MemoryBlock block;
#ifdef MACE_ENABLE_OPENCL
  if (mem_type == MemoryType::GPU_IMAGE) {
    OpenCLBufferType buffer_type = OpenCLBufferType::IN_OUT_CHANNEL;
    if (op_def->type() == "BufferTransform") {
      buffer_type = static_cast<OpenCLBufferType>(
          ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
              *op_def, "buffer_type", OpenCLBufferType::IN_OUT_CHANNEL));
    }
    std::vector<size_t> image_shape;
    if (shape.size() == 2) {
      shape = {shape[0], 1, 1, shape[1]};
    } else {
      MACE_CHECK(shape.size() == 4) << "GPU only support 2D/4D input";
    }
    OpenCLUtil::CalImage2DShape(shape, buffer_type, &image_shape);
    block.set_x(image_shape[0]);
    block.set_y(image_shape[1]);
    return block;
  }
#endif  // MACE_ENABLE_OPENCL
  MACE_UNUSED(mem_type);
  int64_t op_mem_size = std::accumulate(shape.begin(),
                                        shape.end(),
                                        GetEnumTypeSize(dt),
                                        std::multiplies<int64_t>());
  block.set_x(op_mem_size);
  block.set_y(1);
  return block;
}

void MemoryOptimizer::Optimize(
    const mace::OperatorDef *op_def,
    const std::unordered_map<std::string, MemoryType> *mem_types) {
  MACE_LATENCY_LOGGER(2, "Optimize memory");
  if (op_def->output_size() != op_def->output_shape_size()) {
    VLOG(1) << op_def->name()
            << ": the number of output shape "
            << "is not equal to the number of output";
    return;
  }

  auto device = static_cast<DeviceType>(op_def->device_type());
  DataType op_dtype = static_cast<DataType>(ProtoArgHelper::GetOptionalArg(
      *op_def,
      "T",
      static_cast<int>(DT_FLOAT)));
  MACE_CHECK(
      op_def->output_type_size() == 0 ||
          op_def->output_size() == op_def->output_type_size(),
      "operator output size != operator output type size",
      op_def->output_size(),
      op_def->output_type_size());
  DataType dt;

  DataFormat data_format = static_cast<DataFormat>(
      ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *op_def, "data_format",
          static_cast<int>(DataFormat::NONE)));
  int output_size = op_def->output_size();
  for (int i = 0; i < output_size; ++i) {
    if (i < op_def->output_type_size()) {
      dt = op_def->output_type(i);
    } else {
      dt = op_dtype;
    }
    int best_mem_id = -1;
    MemoryType mem_type = MemoryType::CPU_BUFFER;
    if (device == DeviceType::GPU) {
      mem_type = mem_types->at(op_def->output(i));
    }
    MemoryBlock op_mem_block = CreateMemoryBlock(op_def, i, dt, mem_type);
    MemoryBlock best_mem_block;
    if (IsMemoryReuseOp(op_def->type())) {
      if (tensor_mem_map_.count(op_def->input(0)) == 1) {
        best_mem_id = tensor_mem_map_.at(op_def->input(0)).mem_id;
      }
    } else {
      int64_t op_mem_size = op_mem_block.x() * op_mem_block.y();
      int64_t best_added_mem_size = LLONG_MAX;
      int64_t best_wasted_mem_size = LLONG_MAX;

      int64_t old_mem_size = 0, new_mem_size = 0;
      MemoryBlock new_mem_block;
      for (auto idle_mem_id : idle_blocks_) {
        if (mem_blocks_[idle_mem_id].mem_type() == mem_type) {
          if (mem_type == MemoryType::GPU_IMAGE) {
            // GPU Image could reuse memory with same data type only
            if (mem_blocks_[idle_mem_id].data_type() != dt) {
              continue;
            }
            old_mem_size =
                mem_blocks_[idle_mem_id].x() * mem_blocks_[idle_mem_id].y();
            new_mem_block.set_x(std::max<int64_t>(mem_blocks_[idle_mem_id].x(),
                                                  op_mem_block.x()));
            new_mem_block.set_y(std::max<int64_t>(mem_blocks_[idle_mem_id].y(),
                                                  op_mem_block.y()));
            new_mem_size = new_mem_block.x() * new_mem_block.y();
          } else {
            old_mem_size = mem_blocks_[idle_mem_id].x();
            new_mem_size = std::max(op_mem_size, old_mem_size);
            new_mem_block.set_x(new_mem_size);
          }
          int64_t added_mem_size = new_mem_size - old_mem_size;
          int64_t wasted_mem_size = new_mem_size - op_mem_size;
          // minimize add_mem_size; if best_mem_add_size is 0,
          // then minimize waste_mem_size
          if ((best_added_mem_size > 0 && added_mem_size < best_added_mem_size)
              || (best_added_mem_size == 0 &&
                  wasted_mem_size < best_wasted_mem_size)) {
            best_mem_id = idle_mem_id;
            best_added_mem_size = added_mem_size;
            best_wasted_mem_size = wasted_mem_size;
            best_mem_block = new_mem_block;
          }
        }
      }

      if (best_added_mem_size <= op_mem_size) {
        best_mem_block.set_mem_id(best_mem_id);
        best_mem_block.set_data_type(dt);
        best_mem_block.set_mem_type(mem_type);
        mem_blocks_[best_mem_id] = best_mem_block;
        idle_blocks_.erase(best_mem_id);
      } else {
        best_mem_id = static_cast<int>(mem_blocks_.size());
        best_mem_block.set_mem_id(best_mem_id);
        best_mem_block.set_data_type(dt);
        best_mem_block.set_mem_type(mem_type);
        best_mem_block.set_x(op_mem_block.x());
        best_mem_block.set_y(op_mem_block.y());
        mem_blocks_.push_back(best_mem_block);
      }
    }

    if (best_mem_id != -1) {
      if (mem_ref_count_.count(best_mem_id) == 1) {
        mem_ref_count_[best_mem_id] += 1;
      } else {
        mem_ref_count_[best_mem_id] = 1;
      }
      tensor_mem_map_.emplace(op_def->output(i), TensorMemInfo(best_mem_id,
          dt, data_format));
    }
  }

  // de-refer input tensors
  int input_size = op_def->input_size();
  for (int i = 0; i < input_size; ++i) {
    auto &input_name = op_def->input(i);
    if (tensor_ref_count_.count(input_name) == 1) {
      tensor_ref_count_[input_name] -= 1;
      if (tensor_ref_count_.at(input_name) == 0 &&
          tensor_mem_map_.count(input_name) == 1) {
        int mem_id = tensor_mem_map_.at(input_name).mem_id;
        mem_ref_count_[mem_id] -= 1;
        if (mem_ref_count_.at(mem_id) == 0) {
          idle_blocks_.insert(mem_id);
        }
      } else {
        MACE_CHECK(tensor_ref_count_.at(input_name) >= 0,
                   "Reference count of tensor ",
                   input_name,
                   " is ",
                   tensor_ref_count_.at(input_name));
      }
    }
  }
}

const std::vector<MemoryBlock>& MemoryOptimizer::mem_blocks() const {
  return mem_blocks_;
}

const std::unordered_map<std::string, MemoryOptimizer::TensorMemInfo>&
    MemoryOptimizer::tensor_mem_map() const {
  return tensor_mem_map_;
}

std::string MemoryOptimizer::DebugInfo() const {
  auto memory_type_to_str = [](const MemoryType type) -> std::string {
    if (type == MemoryType::CPU_BUFFER) {
      return "CPU_BUFFER";
    } else if (type == MemoryType::GPU_BUFFER) {
      return "GPU_BUFFER";
    } else if (type == MemoryType::GPU_IMAGE) {
      return "GPU_IMAGE";
    } else {
      return "UNKNOWN";
    }
  };
  std::stringstream sstream;
  sstream << "\n";
  size_t block_size = mem_blocks_.size();
  for (size_t i = 0; i < block_size; ++i) {
    sstream << i << " " << memory_type_to_str(mem_blocks_[i].mem_type())
            << " ";
    if (mem_blocks_[i].mem_type() == MemoryType::GPU_IMAGE) {
      sstream << DataTypeToString(mem_blocks_[i].data_type()) << " "
              "[" << mem_blocks_[i].x() << ", " << mem_blocks_[i].y() << "]";
    } else {
      sstream << "[" << mem_blocks_[i].x() << "]";
    }
    sstream << "\n";
  }

  for (auto tuple : tensor_mem_map_) {
    sstream << tuple.first << ": " << tuple.second.mem_id << ", ";
    if (mem_blocks_[tuple.second.mem_id].mem_type() == MemoryType::GPU_IMAGE) {
      sstream << "[" << mem_blocks_[tuple.second.mem_id].x() << ", "
              << mem_blocks_[tuple.second.mem_id].y() << "]";
    } else {
      sstream << "[" << mem_blocks_[tuple.second.mem_id].x() << "]";
    }
    sstream << "\n";
  }

  return sstream.str();
}

}  // namespace mace
