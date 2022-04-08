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

#include "mace/runtimes/qnn/op_builder.h"

#include <functional>

#include "mace/core/proto/arg_helper.h"
#include "third_party/qnn/include/QnnTensor.h"
#include "third_party/qnn/include/QnnTypes.h"

namespace mace {
namespace {
uint32_t GetQnnId() {
  static uint32_t id = 10000;
  return ++id;
}

uint32_t DataTypeSize(Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      return 8;
    default:
      LOG(FATAL) << "Wrong data type: " << static_cast<int>(data_type);
      return 0;
  }
}

Qnn_DataType_t MapToQnnDataType(DataType type, bool params = false) {
  if (params) {
    if (type == DT_UINT8) return QNN_DATATYPE_UINT_8;
    if (type == DT_UINT16) return QNN_DATATYPE_UINT_16;
    if (type == DT_INT32) return QNN_DATATYPE_INT_32;
    if (type == DT_UINT32) return QNN_DATATYPE_UINT_32;
  } else {
    if (type == DT_UINT8) return QNN_DATATYPE_UFIXED_POINT_8;
    if (type == DT_UINT16) return QNN_DATATYPE_UFIXED_POINT_16;
    if (type == DT_INT32) return QNN_DATATYPE_SFIXED_POINT_32;
  }
  if (type != DT_FLOAT) {
    MACE_NOT_IMPLEMENTED;
  }
  return QNN_DATATYPE_FLOAT_32;
}

std::string QnnTensorTypeToString(const Qnn_TensorType_t type) {
  switch (type) {
    case QNN_TENSOR_TYPE_APP_WRITE: return "QNN_TENSOR_TYPE_APP_WRITE";
    case QNN_TENSOR_TYPE_APP_READ: return "QNN_TENSOR_TYPE_APP_READ";
    case QNN_TENSOR_TYPE_APP_READWRITE: return "QNN_TENSOR_TYPE_APP_READWRITE";
    case QNN_TENSOR_TYPE_NATIVE: return "QNN_TENSOR_TYPE_NATIVE";
    case QNN_TENSOR_TYPE_STATIC: return "QNN_TENSOR_TYPE_STATIC";
    case QNN_TENSOR_TYPE_NULL: return "QNN_TENSOR_TYPE_NULL";
    case QNN_TENSOR_TYPE_UNDEFINED: return "QNN_TENSOR_TYPE_UNDEFINED";
    default: return MakeString("UNKNOWN: ", type);
  }
}

std::string QnnDataTypeToString(const Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_INT_8: return "QNN_DATATYPE_INT_8";
    case QNN_DATATYPE_INT_16: return "QNN_DATATYPE_INT_16";
    case QNN_DATATYPE_INT_32: return "QNN_DATATYPE_INT_32";
    case QNN_DATATYPE_INT_64: return "QNN_DATATYPE_INT_64";
    case QNN_DATATYPE_UINT_8: return "QNN_DATATYPE_UINT_8";
    case QNN_DATATYPE_UINT_16: return "QNN_DATATYPE_UINT_16";
    case QNN_DATATYPE_UINT_32: return "QNN_DATATYPE_UINT_32";
    case QNN_DATATYPE_UINT_64: return "QNN_DATATYPE_UINT_64";
    case QNN_DATATYPE_FLOAT_16: return "QNN_DATATYPE_FLOAT_16";
    case QNN_DATATYPE_FLOAT_32: return "QNN_DATATYPE_FLOAT_32";
    case QNN_DATATYPE_SFIXED_POINT_8: return "QNN_DATATYPE_SFIXED_POINT_8";
    case QNN_DATATYPE_SFIXED_POINT_16: return "QNN_DATATYPE_SFIXED_POINT_16";
    case QNN_DATATYPE_SFIXED_POINT_32: return "QNN_DATATYPE_SFIXED_POINT_32";
    case QNN_DATATYPE_UFIXED_POINT_8: return "QNN_DATATYPE_UFIXED_POINT_8";
    case QNN_DATATYPE_UFIXED_POINT_16: return "QNN_DATATYPE_UFIXED_POINT_16";
    case QNN_DATATYPE_UFIXED_POINT_32: return "QNN_DATATYPE_UFIXED_POINT_32";
    case QNN_DATATYPE_BOOL_8: return "QNN_DATATYPE_BOOL_8";
    case QNN_DATATYPE_UNDEFINED: return "QNN_DATATYPE_UNDEFINED";
    default: return MakeString("UNKNOWN: ", type);
  }
}

std::string QnnDefinitionToString(const Qnn_Definition_t type) {
  switch (type) {
    case QNN_DEFINITION_IMPL_GENERATED: return "QNN_DEFINITION_IMPL_GENERATED";
    case QNN_DEFINITION_DEFINED: return "QNN_DEFINITION_DEFINED";
    case QNN_DEFINITION_UNDEFINED: return "QNN_DEFINITION_UNDEFINED";
    default: return MakeString("UNKNOWN: ", type);
  }
}

std::string QnnQuantizationEncodingToString(
    const Qnn_QuantizationEncoding_t type) {
  switch (type) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      return "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET";
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      return "QNN_QUANTIZATION_ENCODING_UNDEFINED";
    default:
      return MakeString("UNKNOWN: ", type);
  }
}


std::string QnnTensorMemTypeToString(const Qnn_TensorMemType_t type) {
  switch (type) {
    case QNN_TENSORMEMTYPE_RAW: return "QNN_TENSORMEMTYPE_RAW";
    case QNN_TENSORMEMTYPE_MEMHANDLE: return "QNN_TENSORMEMTYPE_MEMHANDLE";
    case QNN_TENSORMEMTYPE_UNDEFINED: return "QNN_TENSORMEMTYPE_UNDEFINED";
    default: return MakeString("UNKNOWN: ", type);
  }
}

std::string QnnParamTypeToString(const Qnn_ParamType_t type) {
  switch (type) {
    case QNN_PARAMTYPE_SCALAR: return "QNN_PARAMTYPE_SCALAR";
    case QNN_PARAMTYPE_TENSOR: return "QNN_PARAMTYPE_TENSOR";
    case QNN_PARAMTYPE_UNDEFINED: return "QNN_PARAMTYPE_UNDEFINED";
    default: return MakeString("UNKNOWN: ", type);
  }
}

void PrintTensor(const Qnn_Tensor_t &tensor) {
  VLOG(1) << "Tensor id: " << tensor.id
          << ", type: " << QnnTensorTypeToString(tensor.type)
          << ", data format: " << tensor.dataFormat << ", data type: "
          << QnnDataTypeToString(tensor.dataType) << ", encoding definition: "
          << QnnDefinitionToString(tensor.quantizeParams.encodingDefinition)
          << ", quantization encoding: "
          << QnnQuantizationEncodingToString(
                 tensor.quantizeParams.quantizationEncoding)
          << ", scale: " << tensor.quantizeParams.scaleOffsetEncoding.scale
          << ", offset: " << tensor.quantizeParams.scaleOffsetEncoding.offset
          << ", rank: " << tensor.rank
          << ", mem type: " << QnnTensorMemTypeToString(tensor.memType)
          << ", address: " << tensor.clientBuf.data
          << ", size: " << tensor.clientBuf.dataSize;
  for (uint32_t i = 0; i < tensor.rank; ++i) {
    VLOG(1) << "max dims[" << i << "]: " << tensor.maxDimensions[i]
            << ", cur dims[" << i << "]: " << tensor.currentDimensions[i];
  }
}

void PrintOpConfig(const Qnn_OpConfig_t &op_config) {
  VLOG(1) << op_config.name << " " << op_config.packageName << " "
          << op_config.typeName;
  VLOG(1) << "Params num: " << op_config.numOfParams;
  for (uint32_t i = 0; i < op_config.numOfParams; ++i) {
    Qnn_Param_t &param = op_config.params[i];
    VLOG(1) << param.name
            << ", param type: " << QnnParamTypeToString(param.paramType);
    if (param.paramType == QNN_PARAMTYPE_TENSOR) {
      Qnn_Tensor_t &tensor = param.tensorParam;
      PrintTensor(tensor);
      for (uint32_t i = 0; i < tensor.clientBuf.dataSize / sizeof(uint32_t);
           ++i) {
        VLOG(1) << "value: "
                << (reinterpret_cast<uint32_t *>(tensor.clientBuf.data))[i];
      }
    } else {
      Qnn_Scalar_t scalar = param.scalarParam;
      VLOG(1) << "Scalar type: " << QnnDataTypeToString(scalar.dataType)
              << ", value: " << scalar.int32Value;
    }
  }
  VLOG(1) << "Inputs num: " << op_config.numOfInputs;
  for (uint32_t i = 0; i < op_config.numOfInputs; ++i) {
    Qnn_Tensor_t &tensor = op_config.inputTensors[i];
    PrintTensor(tensor);
  }
  VLOG(1) << "Outputs num: " << op_config.numOfOutputs;
  for (uint32_t i = 0; i < op_config.numOfOutputs; ++i) {
    Qnn_Tensor_t &tensor = op_config.outputTensors[i];
    PrintTensor(tensor);
  }
}
}  // namespace

namespace qnn {
extern void RegisterActivation(OpRegistry *);
extern void RegisterArgMax(OpRegistry *);
extern void RegisterBatchNorm(OpRegistry *);
extern void RegisterCast(OpRegistry *);
extern void RegisterConcat(OpRegistry *);
extern void RegisterConv2D(OpRegistry *);
extern void RegisterDeconv2D(OpRegistry *);
extern void RegisterDepthToSpace(OpRegistry *);
extern void RegisterEltwise(OpRegistry *);
extern void RegisterExpandDims(OpRegistry *);
extern void RegisterFullyConnected(OpRegistry *);
extern void RegisterGather(OpRegistry *);
extern void RegisterInstanceNorm(OpRegistry *);
extern void RegisterMatMul(OpRegistry *);
extern void RegisterMoments(OpRegistry *);
extern void RegisterPack(OpRegistry *);
extern void RegisterPad(OpRegistry *);
extern void RegisterPooling(OpRegistry *);
extern void RegisterQuantize(OpRegistry *);
extern void RegisterReduce(OpRegistry *);
extern void RegisterReshape(OpRegistry *);
extern void RegisterResize(OpRegistry *);
extern void RegisterSoftmax(OpRegistry *);
extern void RegisterSpaceToDepth(OpRegistry *);
extern void RegisterSplit(OpRegistry *);
extern void RegisterSqueeze(OpRegistry *);
extern void RegisterStridedSlice(OpRegistry *);
extern void RegisterTile(OpRegistry *);
extern void RegisterTranspose(OpRegistry *);
void RegisterAllOps(OpRegistry *registry) {
  RegisterActivation(registry);
  RegisterArgMax(registry);
  RegisterBatchNorm(registry);
  RegisterCast(registry);
  RegisterConcat(registry);
  RegisterConv2D(registry);
  RegisterDeconv2D(registry);
  RegisterDepthToSpace(registry);
  RegisterEltwise(registry);
  RegisterExpandDims(registry);
  RegisterFullyConnected(registry);
  RegisterGather(registry);
  RegisterInstanceNorm(registry);
  RegisterMatMul(registry);
  RegisterMoments(registry);
  RegisterPack(registry);
  RegisterPad(registry);
  RegisterPooling(registry);
  RegisterQuantize(registry);
  RegisterReduce(registry);
  RegisterReshape(registry);
  RegisterResize(registry);
  RegisterSoftmax(registry);
  RegisterSpaceToDepth(registry);
  RegisterSplit(registry);
  RegisterSqueeze(registry);
  RegisterStridedSlice(registry);
  RegisterTile(registry);
  RegisterTranspose(registry);
}
}  // namespace qnn

void OpBuilder::AddInput(const std::string &name) {
  AddInput(graph_builder_->GetTensor(name));
}

void OpBuilder::AddOutput(const std::string &name) {
  AddOutput(graph_builder_->GetTensor(name));
}

void OpBuilder::AddTensorParamNotCreat(const char *name,
                               const std::string &tensor_name) {
  Qnn_Param_t param = {
      .paramType = QNN_PARAMTYPE_TENSOR,
      .name = name,
      .tensorParam = graph_builder_->GetTensor(tensor_name)};
  params_.push_back(param);
}

void OpBuilder::AddTensorParam(const char *name,
                               const std::vector<uint32_t> &dims,
                               const void *data,
                               const Qnn_DataType_t data_type) {
  Qnn_Param_t param = {
      .paramType = QNN_PARAMTYPE_TENSOR,
      .name = name,
      .tensorParam = graph_builder_->CreateParamTensor(dims, data, data_type)};
  params_.push_back(param);
}

void OpBuilder::AddScalarParam(const char* name, const Qnn_Scalar_t scalar) {
  Qnn_Param_t param = {
      .paramType = QNN_PARAMTYPE_SCALAR,
      .name = name,
      .scalarParam = scalar};
  params_.push_back(param);
}

Qnn_Tensor_t GraphBuilder::CreateParamTensor(
    const std::vector<uint32_t> &tensor_dims,
    const void *tensor_data,
    const Qnn_DataType_t data_type) {
  Qnn_Tensor_t tensor = {
      .id = GetQnnId(),
      .type = QNN_TENSOR_TYPE_STATIC,
      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
      .dataType = data_type,
      .quantizeParams = {.encodingDefinition = QNN_DEFINITION_UNDEFINED,
                         .quantizationEncoding =
                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                         {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
      .rank = static_cast<uint32_t>(tensor_dims.size()),
      .maxDimensions = const_cast<uint32_t *>(tensor_dims.data()),
      .currentDimensions = const_cast<uint32_t *>(tensor_dims.data()),
      .memType = QNN_TENSORMEMTYPE_RAW,
      {.clientBuf = {
           .data = const_cast<void *>(tensor_data),
           .dataSize = std::accumulate(tensor_dims.begin(), tensor_dims.end(),
                                       DataTypeSize(data_type),
                                       std::multiplies<uint32_t>())}}};
  PrintTensor(tensor);

  Qnn_ErrorHandle_t ret = qnn_function_pointers_->qnnInterface.tensorCreateGraphTensor(graph_,
                                                                                       tensor);
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnTensor_createGraphTensor failed with error: ", ret);
  return tensor;
}

void GraphBuilder::CreateGraphTensor(const std::string &tensor_name,
                                     const uint32_t id,
                                     const Qnn_TensorType_t tensor_type,
                                     const Qnn_DataType_t data_type,
                                     const float scale,
                                     const int32_t zero_point,
                                     const std::vector<uint32_t> &tensor_dims,
                                     const void *tensor_data,
                                     const uint32_t tensor_data_size) {
  if (tensor_map_.count(tensor_name)) {
    return;
  }
  tensor_map_[tensor_name] = TensorInfo(tensor_dims);
  TensorInfo &tensor_info = tensor_map_[tensor_name];
  bool fp32 = (data_type == QNN_DATATYPE_FLOAT_32 ? 1 : 0);
  tensor_info.tensor = {
      .id = (id == 0 ? GetQnnId() : id),
      .type = tensor_type,
      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
      .dataType = data_type,
      .quantizeParams = {.encodingDefinition =
                             (fp32 ? QNN_DEFINITION_UNDEFINED : QNN_DEFINITION_DEFINED),
                         .quantizationEncoding =
                             (fp32 ? QNN_QUANTIZATION_ENCODING_UNDEFINED :
                              QNN_QUANTIZATION_ENCODING_SCALE_OFFSET),
                         {.scaleOffsetEncoding = {.scale = scale,
                                                  .offset = -zero_point}}},
      .rank = static_cast<uint32_t>(tensor_info.shape.size()),
      .maxDimensions = tensor_info.shape.data(),
      .currentDimensions = tensor_info.shape.data(),
      .memType = QNN_TENSORMEMTYPE_RAW,
      {.clientBuf = {.data = const_cast<void *>(tensor_data),
                     .dataSize = tensor_data_size}}};
  VLOG(1) << tensor_name;
  PrintTensor(tensor_info.tensor);

  Qnn_ErrorHandle_t ret =
      qnn_function_pointers_->qnnInterface.tensorCreateGraphTensor(graph_, tensor_info.tensor);
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnTensor_createGraphTensor failed with error: ", ret);
}

void GraphBuilder::AddGraphNode(const OpBuilder &op_builder) {
  Qnn_OpConfig_t op_config = {
      .name = op_builder.GetOpName(),
      .packageName = op_builder.GetPackageName(),
      .typeName = op_builder.GetOpType(),
      .numOfParams = static_cast<uint32_t>(op_builder.GetParams().size()),
      .params = const_cast<Qnn_Param_t *>(op_builder.GetParams().data()),
      .numOfInputs = static_cast<uint32_t>(op_builder.GetInputs().size()),
      .inputTensors = const_cast<Qnn_Tensor_t *>(op_builder.GetInputs().data()),
      .numOfOutputs = static_cast<uint32_t>(op_builder.GetOutputs().size()),
      .outputTensors =
          const_cast<Qnn_Tensor_t *>(op_builder.GetOutputs().data())};
  PrintOpConfig(op_config);

  auto validationStatus = qnn_function_pointers_->qnnInterface.backendValidateOpConfig(op_config);
  MACE_CHECK(validationStatus == QNN_SUCCESS,
             "QnnModel::addNode() validating node failed.");
  Qnn_ErrorHandle_t ret = qnn_function_pointers_->qnnInterface.graphAddNode(graph_, op_config);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnGraph_addNode failed with error: ", ret);
}

void GraphBuilder::AddModelInputs(std::vector<QnnInOutInfo> *infos,
                                  std::vector<Qnn_Tensor_t> *tensors) {
  tensors->resize(net_def_->input_info_size());
  for (int i = 0; i < net_def_->input_info_size(); ++i) {
    const auto &input_info = net_def_->input_info(i);
    std::vector<uint32_t> shape(input_info.dims().begin(),
                                input_info.dims().end());
    std::vector<index_t> mace_shape(input_info.dims().begin(),
                                    input_info.dims().end());

    auto quantized_tensor = make_unique<Tensor>(runtime_, quantized_type_,
                                                runtime_->GetBaseMemoryType());
    quantized_tensor->Resize(mace_shape);
    quantized_tensor->SetScale(input_info.scale());
    quantized_tensor->SetZeroPoint(input_info.zero_point());

    infos->emplace_back(input_info.name(), input_info.data_type(), shape,
                        input_info.scale(), input_info.zero_point(),
                        std::move(quantized_tensor));
    Tensor::MappingGuard input_guard((*infos)[i].quantized_tensor.get());

    Qnn_DataType_t qnn_quantized_type = quantized_type_ == DT_UINT16 ?
                                        QNN_DATATYPE_UFIXED_POINT_16 :
                                        QNN_DATATYPE_UFIXED_POINT_8;
    if (input_info.data_type() == DT_FLOAT) {
      CreateGraphTensor(input_info.name(), 0, QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_FLOAT_32, 0.0f,
                        0, infos->back().shape);
    } else if (input_info.data_type() == DT_INT32) {
      CreateGraphTensor(input_info.name(), 0, QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_INT_32, 0.0f,
                        0, infos->back().shape);
    } else {
      CreateGraphTensor(input_info.name(), 0, QNN_TENSOR_TYPE_APP_WRITE,
                        qnn_quantized_type, input_info.scale(),
                        input_info.zero_point(), infos->back().shape);
    }

    (*tensors)[i] = tensor_map_[input_info.name()].tensor;
  }
}

void GraphBuilder::AddModelOutputs(std::vector<QnnInOutInfo> *infos,
                                   std::vector<Qnn_Tensor_t> *tensors) {
  tensors->resize(net_def_->output_info_size());
  for (int i = 0; i < net_def_->output_info_size(); ++i) {
    const auto &output_info = net_def_->output_info(i);
    std::vector<uint32_t> shape(output_info.dims().begin(),
                                output_info.dims().end());
    std::vector<index_t> mace_shape(output_info.dims().begin(),
                                    output_info.dims().end());

    auto quantized_tensor = make_unique<Tensor>(runtime_, quantized_type_,
                                                runtime_->GetBaseMemoryType());
    quantized_tensor->Resize(mace_shape);
    quantized_tensor->SetScale(output_info.scale());
    quantized_tensor->SetZeroPoint(output_info.zero_point());

    infos->emplace_back(output_info.name(), output_info.data_type(), shape,
                        output_info.scale(), output_info.zero_point(),
                        std::move(quantized_tensor));
    Tensor::MappingGuard output_guard((*infos)[i].quantized_tensor.get());

    Qnn_DataType_t qnn_quantized_type = quantized_type_ == DT_UINT16 ?
                                        QNN_DATATYPE_UFIXED_POINT_16 :
                                        QNN_DATATYPE_UFIXED_POINT_8;

    if (quantized_type_ == DT_UINT16 || quantized_type_ == DT_UINT8) {
      CreateGraphTensor(output_info.name(), 0, QNN_TENSOR_TYPE_APP_READ,
                        qnn_quantized_type, output_info.scale(),
                        output_info.zero_point(), infos->back().shape);
    } else if (output_info.data_type() == DT_FLOAT) {
      CreateGraphTensor(output_info.name(), 0, QNN_TENSOR_TYPE_APP_READ,
                        QNN_DATATYPE_FLOAT_32, 0.0f,
                        0, infos->back().shape);
    } else if (output_info.data_type() == DT_INT32) {
      CreateGraphTensor(output_info.name(), 0, QNN_TENSOR_TYPE_APP_READ,
                        QNN_DATATYPE_INT_32, 0.0f,
                        0, infos->back().shape);
    }

    (*tensors)[i] = tensor_map_[output_info.name()].tensor;
  }
}

void GraphBuilder::AddModelInputsFromOfflineCache(
    std::vector<QnnInOutInfo> *infos,
    std::vector<Qnn_Tensor_t> *tensors,
    const uint32_t *ids) {
  tensors->resize(net_def_->input_info_size());
  for (int i = 0; i < net_def_->input_info_size(); ++i) {
    const auto &input_info = net_def_->input_info(i);
    std::vector<uint32_t> shape(input_info.dims().begin(),
                                input_info.dims().end());
    std::vector<index_t> mace_shape(input_info.dims().begin(),
                                input_info.dims().end());

    auto quantized_tensor = make_unique<Tensor>(runtime_, quantized_type_,
                                                runtime_->GetBaseMemoryType());
    quantized_tensor->Resize(mace_shape);
    quantized_tensor->SetScale(input_info.scale());
    quantized_tensor->SetZeroPoint(input_info.zero_point());

    infos->emplace_back(input_info.name(), input_info.data_type(), shape,
                        input_info.scale(), input_info.zero_point(),
                        std::move(quantized_tensor));
    Tensor::MappingGuard input_guard((*infos)[i].quantized_tensor.get());

    Qnn_DataType_t qnn_quantized_type = quantized_type_ == DT_UINT16 ?
                                        QNN_DATATYPE_UFIXED_POINT_16 :
                                        QNN_DATATYPE_UFIXED_POINT_8;
    Qnn_Tensor_t tensor_info = {
        .id = ids[i],
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = qnn_quantized_type,
        .quantizeParams =
            {.encodingDefinition = QNN_DEFINITION_DEFINED,
             .quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
             {.scaleOffsetEncoding = {.scale = input_info.scale(),
                                      .offset = -input_info.zero_point()}}},
        .rank = static_cast<uint32_t>(input_info.dims_size()),
        .maxDimensions = infos->back().shape.data(),
        .currentDimensions = infos->back().shape.data(),
        .memType = QNN_TENSORMEMTYPE_RAW};

    (*tensors)[i] = tensor_info;
  }
}

void GraphBuilder::AddModelOutputsFromOfflineCache(
    std::vector<QnnInOutInfo> *infos,
    std::vector<Qnn_Tensor_t> *tensors,
    const uint32_t *ids) {
  tensors->resize(net_def_->output_info_size());
  for (int i = 0; i < net_def_->output_info_size(); ++i) {
    const auto &output_info = net_def_->output_info(i);
    std::vector<uint32_t> shape(output_info.dims().begin(),
                                output_info.dims().end());
    std::vector<index_t> mace_shape(output_info.dims().begin(),
                                output_info.dims().end());
    auto quantized_tensor = make_unique<Tensor>(runtime_, quantized_type_,
                                                runtime_->GetBaseMemoryType());
    quantized_tensor->Resize(mace_shape);
    quantized_tensor->SetScale(output_info.scale());
    quantized_tensor->SetZeroPoint(output_info.zero_point());

    infos->emplace_back(output_info.name(), output_info.data_type(), shape,
                        output_info.scale(), output_info.zero_point(),
                        std::move(quantized_tensor));
    Tensor::MappingGuard output_guard((*infos)[i].quantized_tensor.get());

    Qnn_DataType_t qnn_quantized_type = quantized_type_ == DT_UINT16 ?
                                        QNN_DATATYPE_UFIXED_POINT_16 :
                                        QNN_DATATYPE_UFIXED_POINT_8;
    Qnn_Tensor_t tensor_info = {
        .id = ids[i],
        .type = QNN_TENSOR_TYPE_APP_READ,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = qnn_quantized_type,
        .quantizeParams =
            {.encodingDefinition = QNN_DEFINITION_DEFINED,
             .quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
             {.scaleOffsetEncoding = {.scale = output_info.scale(),
                                      .offset = -output_info.zero_point()}}},
        .rank = static_cast<uint32_t>(output_info.dims_size()),
        .maxDimensions = infos->back().shape.data(),
        .currentDimensions = infos->back().shape.data(),
        .memType = QNN_TENSORMEMTYPE_RAW};
    (*tensors)[i] = tensor_info;
  }
}

void GraphBuilder::AddConstTensors(unsigned const char *model_data,
                                   const index_t model_data_size) {
  MACE_CHECK(model_data != nullptr && model_data_size > 0);
  for (const ConstTensor &const_tensor : net_def_->tensors()) {
    unsigned char *tensor_data =
        const_cast<unsigned char *>(model_data + const_tensor.offset());
    index_t tensor_data_len =
        const_tensor.data_size() * GetEnumTypeSize(const_tensor.data_type());
    if (model_data_size >= 0) {
      MACE_CHECK(const_tensor.offset() + tensor_data_len <= model_data_size,
                 "tensor end (", const_tensor.offset() + tensor_data_len,
                 ") should <= ", model_data_size);
    }

    std::vector<uint32_t> tensor_dims(const_tensor.dims().begin(),
                                      const_tensor.dims().end());
    if (quantized_type_ != DT_UINT16 && quantized_type_ != DT_UINT8) {
        CreateGraphTensor(const_tensor.name(), 0, QNN_TENSOR_TYPE_STATIC,
                          MapToQnnDataType(const_tensor.data_type(), true),
                          const_tensor.scale(), const_tensor.zero_point(),
                          tensor_dims, tensor_data, tensor_data_len);
    } else {
        CreateGraphTensor(const_tensor.name(), 0, QNN_TENSOR_TYPE_STATIC,
                          MapToQnnDataType(const_tensor.data_type()),
                          const_tensor.scale(), const_tensor.zero_point(),
                          tensor_dims, tensor_data, tensor_data_len);
    }
  }
}

void GraphBuilder::AddOpsOutputs() {
  for (const auto &op : net_def_->op()) {
    if (op.type() == "Quantize") {
      tensor_map_[op.output(0)] = tensor_map_[op.input(0)];
      VLOG(1) << op.type() << ": " << op.input(0) << " replace "
              << op.output(0) << std::endl;
      continue;
    }
    if (op.type() == "Dequantize") {
      if (tensor_map_.find(op.output(0)) != tensor_map_.end()) {
        tensor_map_[op.input(0)] = tensor_map_[op.output(0)];
        VLOG(1) << op.type() << ": " << op.output(0) << " replace "
                << op.input(0) << std::endl;
      }
      continue;
    }
    for (int i = 0; i < op.output_size(); ++i) {
      auto dt = static_cast<DataType>(
          ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
              op, "T", static_cast<int>(DT_UINT8)));
      if (op.output_type_size() > i) {
        dt = op.output_type(i);
      }
      std::vector<uint32_t> tensor_dims(op.output_shape(i).dims().begin(),
                                        op.output_shape(i).dims().end());
      if (dt == DT_FLOAT || dt == DT_UINT32 || dt == DT_INT32) {
        CreateGraphTensor(op.output(i), 0, QNN_TENSOR_TYPE_NATIVE,
                          MapToQnnDataType(dt, true), 0.0f, 0, tensor_dims);
      } else {
        const auto quantize_info = op.quantize_info(i);
        CreateGraphTensor(op.output(i), 0, QNN_TENSOR_TYPE_NATIVE,
                          MapToQnnDataType(dt), quantize_info.scale(),
                          quantize_info.zero_point(), tensor_dims);
      }
    }
  }
}

void GraphBuilder::AddOps() {
  qnn::OpRegistry registry;
  qnn::RegisterAllOps(&registry);
  for (const OperatorDef &op : net_def_->op()) {
    if (op.type() == "Dequantize" || op.type() == "Quantize") {
      continue;
    }

    VLOG(1) << "############################################# "
            << "Building op " << op.name() << " " << op.type()
            << " #############################################";

    qnn::OpCreator &creator = registry.GetOpCreator(op.type());
    auto op_builder = creator(this);
    op_builder->BuildOp(op, quantized_type_);
    AddGraphNode(*op_builder);
  }
}
}  // namespace mace
