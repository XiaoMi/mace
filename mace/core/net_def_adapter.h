// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_NET_DEF_ADAPTER_H_
#define MACE_CORE_NET_DEF_ADAPTER_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"
#include "mace/port/port.h"
#include "mace/core/operator.h"
#include "mace/core/net_optimizer.h"

namespace mace {

class OpRegistryBase;
class Workspace;
class Device;

///////////////////////////////////////////////////////////////////////////////
///                               Conventions
///
/// 1. For the Ops ran with data format(like Conv2D),
///    The inputs and outputs are DataFormat::NCHW if ran on CPU
///    with float data type.
///    while the inputs and outputs are DataFormat::NHWC for
///    other situation(ran on GPU, quantization, DSP)
///
/// 2. Op with DataFormat::AUTO stands for inputs must have
///    fixed format (NHWC or NCHW), determined at runtime.
///
/// 3. if Op with DataFormat::AUTO, the arguments of this op
///    is formatted to NHWC.
///////////////////////////////////////////////////////////////////////////////
class NetDefAdapter {
 public:
  NetDefAdapter(const OpRegistryBase *op_registry,
                const Workspace *ws);
  // Adapt original net_def to a better net.
  // 1. Adapt device: choose best device for every op in the net.
  // 2. Adapt data type: Add data type related transform ops
  //                     for mixing precision.
  // 3. Adapt data format: confirm data format of every op
  //                       and add transpose if necessary.
  // 4. Adapt memory type: Add BufferTransform if necessary
  //                       for transforming memory type between ops.
  MaceStatus AdaptNetDef(
      const NetDef *net_def,
      Device *target_device,
      NetDef *target_net_def);

 public:
  NetDefAdapter(const NetDefAdapter&) = delete;
  NetDefAdapter(const NetDefAdapter&&) = delete;
  NetDefAdapter &operator=(const NetDefAdapter &) = delete;
  NetDefAdapter &operator=(const NetDefAdapter &&) = delete;

 private:
  struct InternalOutputInfo {
    InternalOutputInfo(const MemoryType mem_type,
                       const DataType dtype,
                       const DataFormat data_format,
                       const std::vector<index_t> &shape,
                       int op_idx)
        : mem_type(mem_type), dtype(dtype), data_format(data_format),
          shape(shape), op_idx(op_idx) {}

    MemoryType mem_type;
    DataType dtype;
    DataFormat data_format;
    std::vector<index_t> shape;  // tensor shape
    int op_idx;  // operation which generate the tensor
  };

  typedef std::unordered_map<std::string, InternalOutputInfo> TensorInfoMap;
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;

 private:
  MaceStatus AdaptDevice(OpConditionContext *context,
                         Device *target_device,
                         Device *cpu_device,
                         const TensorInfoMap &output_map,
                         const NetDef *net_def,
                         OperatorDef *op);
  MaceStatus AdaptDataType(OpConditionContext *context,
                           OperatorDef *op);
  MaceStatus AdaptDataFormat(
      OpConditionContext *context,
      OperatorDef *op,
      bool is_quantized_model,
      TensorInfoMap *output_map,
      TensorShapeMap *tensor_shape_map,
      std::unordered_set<std::string> *transformed_set,
      DataFormat *op_output_df,
      NetDef *target_net_def);

  MaceStatus AdaptMemoryType(
      OpConditionContext *context,
      OperatorDef *op_def,
      TensorInfoMap *output_map,
      TensorShapeMap *tensor_shape_map,
      std::unordered_set<std::string> *transformed_set,
      MemoryType *op_output_mem_types,
      NetDef *target_net_def);

  std::string DebugString(const NetDef *net_def);

 private:
  const OpRegistryBase *op_registry_;
  const Workspace *ws_;
  NetOptimizer net_optimizer_;
};

}  // namespace mace
#endif  // MACE_CORE_NET_DEF_ADAPTER_H_
