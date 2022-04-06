// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_QNN_OP_BUILDER_H_
#define MACE_RUNTIMES_QNN_OP_BUILDER_H_

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "mace/core/tensor.h"
#include "mace/runtimes/qnn/common.h"

#include "third_party/qnn/include/HTP/QnnDspGraph.h"
#include "third_party/qnn/include/QnnGraph.h"
#include "third_party/qnn/include/QnnOpDef.h"
#include "third_party/qnn/include/QnnTensor.h"
#include "third_party/qnn/include/QnnTypes.h"

namespace mace {

struct TensorInfo {
  std::vector<uint32_t> shape;
  Qnn_Tensor_t tensor;

  TensorInfo() {}
  explicit TensorInfo(const std::vector<uint32_t> &shape) : shape(shape) {}
  explicit TensorInfo(const std::vector<uint32_t> &shape, Qnn_Tensor_t tensor)
      : shape(shape), tensor(tensor) {}
};
typedef std::unordered_map<std::string, TensorInfo> TensorInfoMap;

class GraphBuilder;

class OpBuilder {
 public:
  explicit OpBuilder(GraphBuilder *graph_builder)
      : graph_builder_(graph_builder),
        op_package_name_(QNN_OP_PACKAGE_NAME_QTI_AISW) {}
  virtual ~OpBuilder() = default;

  virtual MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) = 0;

  const std::vector<Qnn_Param_t> &GetParams() const { return params_; }
  const std::vector<Qnn_Tensor_t> &GetInputs() const { return inputs_; }
  const std::vector<Qnn_Tensor_t> &GetOutputs() const { return outputs_; }
  const char *GetOpName() const { return op_name_; }
  const char *GetOpType() const { return op_type_; }
  const char *GetPackageName() const { return op_package_name_; }

 protected:
  void AddTensorParam(const char *name,
                      const std::vector<uint32_t> &dims,
                      const void *data,
                      const Qnn_DataType_t data_type = QNN_DATATYPE_UINT_32);
  void AddTensorParamNotCreat(const char *name,
                      const std::string &tensor_name);
  void AddScalarParam(const char* name, const Qnn_Scalar_t scalar);
  void AddInput(const Qnn_Tensor_t &tensor) { inputs_.push_back(tensor); }
  void AddInput(const std::string &name);
  void AddOutput(const Qnn_Tensor_t &tensor) { outputs_.push_back(tensor); }
  void AddOutput(const std::string &name);
  void SetOpType(const char *type) { op_type_ = type; }
  void SetOpName(const char *name) { op_name_ = name; }
  void SetOpPackageName(const char *name) { op_package_name_ = name; }

  GraphBuilder *graph_builder_ = nullptr;
  const char *op_type_ = nullptr;
  const char *op_name_ = nullptr;
  const char *op_package_name_ = nullptr;
  std::vector<Qnn_Param_t> params_;
  std::vector<Qnn_Tensor_t> inputs_;
  std::vector<Qnn_Tensor_t> outputs_;
};

class GraphBuilder {
 public:
  GraphBuilder() {}

  void Init(const NetDef *net_def,
            Qnn_GraphHandle_t graph,
            Runtime *runtime,
            DataType quantized_type,
            QnnFunctionPointers* qnn_function_pointers) {
    net_def_ = net_def;
    graph_ = graph;
    runtime_ = runtime;
    quantized_type_ = quantized_type;
    qnn_function_pointers_ = qnn_function_pointers;
  }
  Qnn_Tensor_t CreateParamTensor(
      const std::vector<uint32_t> &tensor_dims,
      const void *tensor_data,
      const Qnn_DataType_t data_type = QNN_DATATYPE_UINT_32);

  void CreateGraphTensor(const std::string &tensor_name,
                         const uint32_t id,
                         const Qnn_TensorType_t tensor_type,
                         const Qnn_DataType_t data_type,
                         const float scale,
                         const int32_t zero_point,
                         const std::vector<uint32_t> &tensor_dims,
                         const void *tensor_data = nullptr,
                         const uint32_t tensor_data_size = 0);

  void AddGraphNode(const OpBuilder &op_builder);
  void SetGraphHandle(Qnn_GraphHandle_t graph) { graph_ = graph; }
  void AddConstTensors(unsigned const char *model_data,
                       const index_t model_data_size);
  void AddModelInputs(std::vector<QnnInOutInfo> *infos,
                      std::vector<Qnn_Tensor_t> *tensors);
  void AddModelOutputs(std::vector<QnnInOutInfo> *infos,
                       std::vector<Qnn_Tensor_t> *tensors);
  void AddModelInputsFromOfflineCache(std::vector<QnnInOutInfo> *infos,
                                      std::vector<Qnn_Tensor_t> *tensors,
                                      const uint32_t *ids);
  void AddModelOutputsFromOfflineCache(std::vector<QnnInOutInfo> *infos,
                                       std::vector<Qnn_Tensor_t> *tensors,
                                       const uint32_t *ids);
  void AddOpsOutputs();
  void AddOps();

  const Qnn_Tensor_t &GetTensor(const std::string &name) {
    return tensor_map_[name].tensor;
  }
  const std::vector<uint32_t> &GetTensorShape(const std::string &name) {
    MACE_CHECK(tensor_map_.count(name) > 0);
    return tensor_map_[name].shape;
  }
  float GetTensorScale(const std::string& name) {
    return tensor_map_[name].tensor.quantizeParams.scaleOffsetEncoding.scale;
  }
  int32_t GetTensorOffset(const std::string& name) {
    return tensor_map_[name].tensor.quantizeParams.scaleOffsetEncoding.offset;
  }

  Qnn_GraphHandle_t GetGraphHandle() { return graph_; }

 private:
  const NetDef *net_def_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  TensorInfoMap tensor_map_;
  Runtime* runtime_;
  DataType quantized_type_;
  QnnFunctionPointers* qnn_function_pointers_;
};

namespace qnn {
typedef std::function<std::unique_ptr<OpBuilder>(GraphBuilder *)> OpCreator;
class OpRegistry {
 public:
  OpRegistry() = default;
  virtual ~OpRegistry() = default;
  template <class DerivedType>
  static std::unique_ptr<OpBuilder> DefaultCreator(
      GraphBuilder *graph_builder) {
    return make_unique<DerivedType>(graph_builder);
  }
  void Register(const std::string &op_type, OpCreator creator) {
    VLOG(3) << "Registering: " << op_type;
    MACE_CHECK(creators.count(op_type) == 0,
               "Op type already registered: ", op_type);
    creators[op_type] = std::move(creator);
  }
  OpCreator &GetOpCreator(const std::string &op_type) {
    MACE_CHECK(creators.count(op_type) > 0,
               "Op type not registered: ", op_type);
    return creators[op_type];
  }
  std::unordered_map<std::string, OpCreator> creators;
};

#define QNN_REGISTER_OP(op_registry, op_type, class_name) \
  op_registry->Register(op_type, OpRegistry::DefaultCreator<class_name>)

void RegisterAllOps(OpRegistry *registry);
}  // namespace qnn
}  // namespace mace

#endif  // MACE_RUNTIMES_QNN_OP_BUILDER_H_
