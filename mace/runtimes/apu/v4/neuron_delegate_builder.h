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

#ifndef MACE_RUNTIMES_APU_V4_NEURON_DELEGATE_BUILDER_H_
#define MACE_RUNTIMES_APU_V4_NEURON_DELEGATE_BUILDER_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "mace/core/runtime/runtime.h"
#include "mace/proto/mace.pb.h"
#include "mace/runtimes/apu/v4/neuron_implementation.h"
#include "third_party/apu/android_R2/neuron_types.h"

#define APU_TENSOR_MAX_DIMS 4

namespace mace {

struct MaceTensorInfo {
  std::string name;
  uint8_t* buf;
  std::vector<int64_t> shape;
  int size;
  int byte_per_element;
  float scale;
  int zero_point;
  DataType data_type;
};

// Track tensor indices to Neuron tensor indices mapping.
class OperandMapping {
 public:
  // Given a TFLite index return the Neuron index. If it doesn't exist
  // return -1.
  int mace_index_to_neuron(int node_id) const {
    int size = mace_tensor_to_neuron_tensor_.size();
    if (node_id >= 0 && node_id < size)
      return mace_tensor_to_neuron_tensor_[node_id];
    else
      return -1;
  }

  // Neuron uses non tensor operands instead of structs. This creates one
  // and returns the index. It uses a std::vector and resizes it as needed
  // keeping -1 to unmapped values. Intermediate tensors likely will not
  // be mapped.
  int add_new_non_tensor_operand() {
    return next_neuron_tensor_index_++;
  }

  // Add a new mapping from `node_id` and return the Neuron tensor
  // index.
  int add_new_neuron_tensor_index(int node_id) {
    int size = mace_tensor_to_neuron_tensor_.size();
    if (node_id >= size) {
      mace_tensor_to_neuron_tensor_.resize(node_id + 1, -1);
    }
    const int new_tensor_index = next_neuron_tensor_index_++;
    mace_tensor_to_neuron_tensor_[node_id] = new_tensor_index;
    return new_tensor_index;
  }


 private:
  // Next index of neuron tensor
  int next_neuron_tensor_index_ = 0;

  // Mapping from mace node_id. Use a std::vector for speed and code size
  // rather than a map.
  std::vector<int> mace_tensor_to_neuron_tensor_;
};




// Abstract builder for building an op in the Neuron graph. This handles
// the disparity between TFLite and Neuron operand types. Neuron has
// singular operands for both tensors and parameters, and TFLite separates the
// two.
class NeuronOpBuilder {
 public:
  NeuronOpBuilder(const NeuronApi* neuronapi, const NetDef* net_def,
                  unsigned const char *model_data, NeuronModel* nn_model)
      : neuronapi_(neuronapi),
        net_def_(net_def),
        model_data_(model_data),
        nn_model_(nn_model) {}

  void AddScalarBoolOperand(bool value) {
    AddScalarOperand<bool>(value, NEURON_BOOL);
  }

  void AddScalarInt32Operand(int32_t value) {
    AddScalarOperand<int32_t>(value, NEURON_INT32);
  }

  void AddScalarFloat32Operand(float value) {
    AddScalarOperand<float>(value, NEURON_FLOAT32);
  }

  void AddVectorInt32Operand(const int32_t* values, uint32_t num_values) {
    AddVectorOperand<int32_t>(values, num_values, NEURON_TENSOR_INT32,
                              /*scale=*/0.f, /*zero_point=*/0);
  }

  void AddArrayInt32Operand(const int32_t* values,
                            uint32_t tensor_rank, uint32_t* tensor_dims) {
    AddArrayOperand<int32_t>(values, tensor_rank, tensor_dims,
                             NEURON_TENSOR_INT32,
                             /*scale=*/0.f, /*zero_point=*/0);
  }

  template <typename T>
  void AddScalarOperand(T value, int32_t nn_type) {
    NeuronOperandType operand_type{.type = nn_type};
    const int neuron_index = operand_mapping_.add_new_non_tensor_operand();
    neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
    neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_index,
                                            &value, sizeof(T));
    augmented_inputs_.push_back(neuron_index);
  }

  template <typename T>
  void AddVectorOperand(const T* values, uint32_t num_values,
                        int32_t nn_type, float scale, int32_t zero_point) {
    NeuronOperandType operand_type{.type = nn_type,
                                   .dimensionCount = 1,
                                   .dimensions = &num_values,
                                   .scale = scale,
                                   .zeroPoint = zero_point};

    neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
    const int neuron_index = operand_mapping_.add_new_non_tensor_operand();
    neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_index, values,
                                            sizeof(T) * num_values);
    augmented_inputs_.push_back(neuron_index);
  }

  template <typename T>
  void AddVectorOperand(const T* values, uint32_t num_values,
                        int32_t nn_type) {
    AddVectorOperand(values, num_values, nn_type,
                     /*scale=*/0.f, /*zero_point=*/0);
  }

  template <typename T>
  void AddArrayOperand(const T* values, uint32_t tensor_rank,
                       uint32_t* tensor_dims,
                       int32_t nn_type, float scale,
                       int32_t zero_point) {
    NeuronOperandType operand_type{.type = nn_type,
                                   .dimensionCount = tensor_rank,
                                   .dimensions = tensor_dims,
                                   .scale = scale,
                                   .zeroPoint = zero_point};

    neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
    const int neuron_index = operand_mapping_.add_new_non_tensor_operand();
    int num_values = 1;
    for (auto i = 0 ; (unsigned)i < tensor_rank; i++) {
      num_values *= tensor_dims[i];
    }
    neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_index, values,
                                                sizeof(T) * num_values);
    augmented_inputs_.push_back(neuron_index);
  }

  template <typename T>
  void AddArrayOperand(const T* values, uint32_t tensor_rank,
                                uint32_t tensor_dims,
                                int32_t nn_type) {
    AddArrayOperand(values, tensor_rank, tensor_dims, nn_type,
                           /*scale=*/0.f, /*zero_point=*/0);
  }
  size_t GetTotalInputByteSize() {
    return total_input_byte_size_;
  }
  size_t GetTotalOutputByteSize() {
    return total_output_byte_size_;
  }
  std::vector<MaceTensorInfo> GetInputInfos() {
    return input_infos_;
  }
  std::vector<MaceTensorInfo> GetOutputInfos() {
    return output_infos_;
  }

  bool SetInputAndOutputImp() {
    // parse model argument
    int apu_dt = -1;
    for (auto arg : net_def_->arg()) {
      if (arg.name().compare("apu_data_type") == 0) {
        apu_dt = arg.i();
      }
    }
    // input tensors
    total_input_byte_size_ = 0;
    for (auto input_info : net_def_->input_info()) {
      // Parameters needed for new type.
      float scale = 0.0f;
      int32_t zeroPoint = 0;
      MaceTensorInfo info;
      scale = input_info.has_scale() ? input_info.scale() : -1.0f;
      zeroPoint = input_info.has_zero_point() ? input_info.zero_point() : 0;
      int byte_per_element = GetByteNum(static_cast<DataType>(apu_dt));
      info.size = 1;
      for (auto i = 0 ; i < input_info.dims_size() ; i++) {
        info.size *= input_info.dims(i);
        info.shape.push_back(input_info.dims(i));
      }
      total_input_byte_size_ += info.size * byte_per_element;
      info.byte_per_element = byte_per_element;
      info.data_type = static_cast<DataType>(apu_dt);
      info.scale = scale;
      info.zero_point = zeroPoint;
      info.name = input_info.name();
      input_infos_.push_back(info);
    }
    // output tensors
    total_output_byte_size_ = 0;
    for (auto output_info : net_def_->output_info()) {
      // Parameters needed for new type.
      MaceTensorInfo info;
      int byte_per_element = GetByteNum(static_cast<DataType>(apu_dt));
      info.size = 1;
      for (auto i = 0 ; i < output_info.dims_size() ; i++) {
        info.size *= output_info.dims(i);
        info.shape.push_back(output_info.dims(i));
      }
      total_output_byte_size_ += info.size * byte_per_element;
      info.byte_per_element = byte_per_element;
      info.data_type = static_cast<DataType>(apu_dt);

      for (auto op_def : net_def_->op()) {
        if (output_info.name() == op_def.output(0)) {
          if (info.data_type == DT_UINT8 ||info.data_type == DT_INT16) {
            info.scale = op_def.quantize_info(0).scale();
            info.zero_point = op_def.quantize_info(0).zero_point();
          } else {
            info.scale = 0.0;
            info.zero_point = 0;
          }
        }
      }
      info.name = output_info.name();
      output_infos_.push_back(info);
    }
    return true;
  }

  bool AddOpsAndTensorsImp(std::vector<int32_t*> *int32_buffers_) {
    // parse model argument
    int const_data_num = 0;
    int apu_dt = -1;
    for (auto arg : net_def_->arg()) {
      if (arg.name().compare("const_data_num") == 0) {
        const_data_num = arg.i();
      } else if (arg.name().compare("apu_data_type") == 0) {
        apu_dt = arg.i();
      }
    }
    // const tensors
    for (auto const_tensor : net_def_->tensors()) {
      int32_t nn_type = 0;
      float scale = 0.0f;
      int32_t zeroPoint = 0;
      int node_id = const_tensor.node_id();
      if (node_id >= const_data_num) {
        break;
      }
      nn_type = MapToApuDataType(const_tensor.data_type());
      scale = const_tensor.has_scale() ? const_tensor.scale() : 0.0f;
      zeroPoint = const_tensor.has_zero_point() ? const_tensor.zero_point() : 0;
      uint32_t tensor_rank = static_cast<uint32_t>(const_tensor.dims_size());
      uint32_t *tensor_dims = nullptr;
      if (tensor_rank > 0) {
        tensor_dims = new uint32_t[tensor_rank];
        for (auto i = 0 ; i < const_tensor.dims_size() ; i++) {
          tensor_dims[i] = const_tensor.dims(i);
        }
      }
      MACE_CHECK(tensor_rank <= APU_TENSOR_MAX_DIMS,
                 "tensor dimension size not supported");
      NeuronOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                     scale, zeroPoint};
      neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
      int byte_per_element = GetByteNum(const_tensor.data_type());
      if (byte_per_element == -1) {
        return false;
      }
      int data_len = const_tensor.data_size() * byte_per_element;
      int neuron_tensor_index = operand_mapping_.mace_index_to_neuron(node_id);
      if (neuron_tensor_index == -1) {
        // Allocate a new tensor index
        neuron_tensor_index =
            operand_mapping_.add_new_neuron_tensor_index(node_id);
      }
      neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_tensor_index,
                  const_cast<unsigned char*>(model_data_
                  + const_tensor.offset()), data_len);
      if (tensor_dims != nullptr) {
        delete [] tensor_dims;
      }
    }
    // input tensors
    total_input_byte_size_ = 0;
    for (auto input_info : net_def_->input_info()) {
      // Parameters needed for new type.
      int32_t nn_type = MapToApuDataType(static_cast<DataType>(apu_dt));
      float scale = 0.0f;
      int32_t zeroPoint = 0;
      MaceTensorInfo info;
      scale = input_info.has_scale() ? input_info.scale() : -1.0f;
      zeroPoint = input_info.has_zero_point() ? input_info.zero_point() : 0;
      uint32_t tensor_rank = static_cast<uint32_t>(input_info.dims_size());
      uint32_t *tensor_dims = nullptr;
      int byte_per_element = GetByteNum(static_cast<DataType>(apu_dt));
      info.size = 1;
      if (tensor_rank > 0) {
        tensor_dims = new uint32_t[tensor_rank];
        for (auto i = 0 ; i < input_info.dims_size() ; i++) {
          tensor_dims[i] = input_info.dims(i);
          info.size *= input_info.dims(i);
          info.shape.push_back(input_info.dims(i));
        }
      }
      total_input_byte_size_ += info.size * byte_per_element;
      info.byte_per_element = byte_per_element;
      info.data_type = static_cast<DataType>(apu_dt);
      info.scale = scale;
      info.zero_point = zeroPoint;
      info.name = input_info.name();
      input_infos_.push_back(info);
      int node_id = input_info.node_id();
      NeuronOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                     scale, zeroPoint};
      neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
      if (tensor_dims != nullptr) {
        delete [] tensor_dims;
      }
      int neuron_tensor_index = operand_mapping_.mace_index_to_neuron(node_id);
      if (neuron_tensor_index == -1) {
        // Allocate a new tensor index
        neuron_tensor_index =
            operand_mapping_.add_new_neuron_tensor_index(node_id);
      }
      modelInputIndexes_.push_back(neuron_tensor_index);
    }
    // operators
    for (auto op_def : net_def_->op()) {
      int32_t nn_type = 0;
      float scale = 0.0f;
      int32_t zeroPoint = 0;
      if (op_def.output_type(0) == DT_UINT8 ||
          op_def.output_type(0) == DT_INT16) {
        scale = op_def.quantize_info(0).scale();
        zeroPoint = op_def.quantize_info(0).zero_point();
      } else {
        scale = 0.0f;
        zeroPoint = 0;
      }
      nn_type = MapToApuDataType(op_def.output_type(0));
      uint32_t tensor_rank =
          static_cast<uint32_t>(op_def.output_shape(0).dims_size());
      uint32_t *tensor_dims = nullptr;
      if (tensor_rank > 0) {
        tensor_dims = new uint32_t[tensor_rank];
        for (auto i = 0 ; i < op_def.output_shape(0).dims_size() ; i++) {
          tensor_dims[i] = op_def.output_shape(0).dims(i);
        }
      }
      NeuronOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                     scale, zeroPoint};
      // Add operation
      int input_size = OpInputNum(op_def);
      for (auto i = 0 ; i < input_size ; i++) {
        int node_id = op_def.node_input(i).node_id();
        int neuron_tensor_index =
            operand_mapping_.mace_index_to_neuron(node_id);
        if (neuron_tensor_index == -1) {
          // Allocate a new tensor index
          neuron_tensor_index =
              operand_mapping_.add_new_neuron_tensor_index(node_id);
        }
        augmented_inputs_.push_back(neuron_tensor_index);
      }
      NeuronOperationType nn_op_type;
      OpMap(op_def, &nn_op_type, int32_buffers_);
      neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type);
      if (tensor_dims != nullptr) {
        delete [] tensor_dims;
      }
      int node_id = op_def.node_id();
      int neuron_tensor_index = operand_mapping_.mace_index_to_neuron(node_id);
      if (neuron_tensor_index == -1) {
        // Allocate a new tensor index
        neuron_tensor_index =
            operand_mapping_.add_new_neuron_tensor_index(node_id);
      }
      augmented_outputs_.push_back(neuron_tensor_index);
      FinalizeAddOperation(nn_op_type);
    }
    // output tensors
    total_output_byte_size_ = 0;
    for (auto output_info : net_def_->output_info()) {
      int neuron_output_index =
          operand_mapping_.mace_index_to_neuron(output_info.node_id());
      if (neuron_output_index == -1) {
        LOG(ERROR) << "Output tensor not found";
        return false;
      }
      modelOutputIndexes_.push_back(neuron_output_index);
      // Parameters needed for new type.
      MaceTensorInfo info;
      int byte_per_element = GetByteNum(static_cast<DataType>(apu_dt));
      info.size = 1;
      for (auto i = 0 ; i < output_info.dims_size() ; i++) {
        info.size *= output_info.dims(i);
        info.shape.push_back(output_info.dims(i));
      }
      total_output_byte_size_ += info.size * byte_per_element;
      info.byte_per_element = byte_per_element;
      info.data_type = static_cast<DataType>(apu_dt);

      for (auto op_def : net_def_->op()) {
        if (output_info.name() == op_def.output(0)) {
          if (info.data_type == DT_UINT8 ||info.data_type == DT_INT16) {
            info.scale = op_def.quantize_info(0).scale();
            info.zero_point = op_def.quantize_info(0).zero_point();
          } else {
            info.scale = 0.0;
            info.zero_point = 0;
          }
        }
      }
      info.name = output_info.name();
      output_infos_.push_back(info);
    }
    // Tell Neuron to declare inputs/outputs
    neuronapi_->NeuronModel_identifyInputsAndOutputs(
          nn_model_, modelInputIndexes_.size(),
          modelInputIndexes_.data(), modelOutputIndexes_.size(),
          modelOutputIndexes_.data());

    neuronapi_->NeuronModel_finish(nn_model_);
    return true;
  }

  // Finish emitting the op (of type `type`) into the Neuron.
  void FinalizeAddOperation(NeuronOperationType type) {
    // Actually add a Neuron operation
    neuronapi_->NeuronModel_addOperation(
        nn_model_, type, static_cast<uint32_t>(augmented_inputs_.size()),
        augmented_inputs_.data(),
        static_cast<uint32_t>(augmented_outputs_.size()),
        augmented_outputs_.data());
    augmented_inputs_.clear();
    augmented_outputs_.clear();
  }

 private:
  // Access to Neuron.
  const NeuronApi* const neuronapi_;
  const NetDef* net_def_;
  // Tracks relationship between indices.
  OperandMapping operand_mapping_;
  unsigned const char *model_data_;
  // The Neuron model.
  NeuronModel* const nn_model_;

  std::vector<int32_t*> int32_buffers_;
  // Inputs and outputs for the current op. These are augmented in the sense
  // that Neuron uses operands for all arguments, not just tensors, unlike
  // MACE.
  std::vector<uint32_t> augmented_inputs_;
  std::vector<uint32_t> augmented_outputs_;
  // Model input/output
  std::vector<MaceTensorInfo> input_infos_;
  std::vector<MaceTensorInfo> output_infos_;
  std::vector<uint32_t> modelInputIndexes_;
  std::vector<uint32_t> modelOutputIndexes_;

  size_t total_input_byte_size_;
  size_t total_output_byte_size_;

  int32_t MapToApuDataType(DataType mace_type) {
    switch (mace_type) {
      case DT_FLOAT:
        return NEURON_TENSOR_FLOAT32;
      case DT_INT32:
        return NEURON_TENSOR_INT32;
      case DT_HALF:
        return NEURON_TENSOR_FLOAT16;
      case DT_FLOAT16:
        return NEURON_TENSOR_FLOAT16;
      case DT_UINT8:
        return NEURON_TENSOR_QUANT8_ASYMM;
      case DT_INT16:
        return NEURON_TENSOR_QUANT16_SYMM;
      default:
        break;
    }
    LOG(ERROR) << "Unsupported APU data type";
    return -1;
  }

  int OpInputNum(OperatorDef op_def) {
    std::string type = op_def.type();
    int input_num = 0;
    if (type == "Activation") {
      input_num = 1;
    } else if (type == "Concat") {
      input_num = op_def.node_input_size() - 1;
    } else if (type == "Conv2D") {
      input_num = 3;
    } else if (type == "DepthwiseConv2d") {
      input_num = 3;
    } else if (type == "Eltwise") {
      input_num = 2;
    } else if (type == "FullyConnected") {
      input_num = 3;
    } else if (type == "Pad") {
      input_num = 1;
    } else if (type == "Pooling") {
      input_num = 1;
    } else if (type == "PRelu") {
      input_num = 2;
    } else if (type == "Reduce") {
      input_num = 1;
    } else if (type == "Reshape") {
      input_num = 1;
    } else if (type == "ResizeBilinear") {
      input_num = 1;
    } else if (type == "Softmax") {
      input_num = 1;
    } else if (type == "Deconv2D") {
      input_num = 2;
    } else if (type == "SpaceToDepth") {
      input_num = 1;
    } else if (type == "DepthToSpace") {
      input_num = 1;
    } else {
    }
    return input_num;
  }


  bool OpMap(OperatorDef op_def, NeuronOperationType* nn_op_type,
             std::vector<int32_t*> *int32_buffers_) {
    std::string type = op_def.type();
    // The argument order in each operators should be fixed.
    if (type == "Activation") {
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      if (activation.s() == "RELU") {
        *nn_op_type = NEURON_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        *nn_op_type = NEURON_RELU6;
      } else if (activation.s() == "SIGMOID") {
        *nn_op_type = NEURON_LOGISTIC;
      } else if (activation.s() == "TANH") {
        *nn_op_type = NEURON_TANH;
      } else {
        LOG(ERROR) << "Unsupport mace activation mode";
        return false;
      }
    } else if (type == "Concat") {
      Argument axis = get_arg(op_def, "axis");
      AddScalarInt32Operand(axis.i());
      *nn_op_type = NEURON_CONCATENATION;
    } else if (type == "Conv2D") {
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      AddScalarBoolOperand(false);  // Use NHWC format
      Argument dilations_values = get_arg(op_def, "dilations");
      AddScalarInt32Operand(dilations_values.ints(0));  // dilation width
      AddScalarInt32Operand(dilations_values.ints(1));  // dilation height
      *nn_op_type = NEURON_CONV_2D;
    } else if (type == "DepthwiseConv2d") {
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      for (auto const_tensor : net_def_->tensors()) {
        if (op_def.input(1) == const_tensor.name()) {
          AddScalarInt32Operand(const_tensor.dims(0));  // depth_multiplier
          break;
        }
      }
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      AddScalarBoolOperand(false);  // Use NHWC format
      Argument dilations_values = get_arg(op_def, "dilations");
      AddScalarInt32Operand(dilations_values.ints(0));  // dilation width
      AddScalarInt32Operand(dilations_values.ints(1));  // dilation height
      *nn_op_type = NEURON_DEPTHWISE_CONV_2D;
    } else if (type == "Eltwise") {
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      Argument type = get_arg(op_def, "type");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      *nn_op_type = NEURON_ADD;
      if (type.i() == 0) {
        *nn_op_type = NEURON_ADD;
      } else if (type.i() == 2) {
        *nn_op_type = NEURON_MUL;
      } else {
        LOG(ERROR) << "Unsupport mace eltwise mode";
        return false;
      }
    } else if (type == "FullyConnected") {
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      *nn_op_type = NEURON_FULLY_CONNECTED;
    } else if (type == "Pad") {
      Argument paddings = get_arg(op_def, "paddings");
      const uint32_t kTensorRank = 2;
      uint32_t tensor_dims[kTensorRank];
      tensor_dims[0] = paddings.ints_size() / 2;
      tensor_dims[1] = 2;
      int size = tensor_dims[0] * tensor_dims[1];
      int32_t* padding_value = new int32_t[size];
      int32_buffers_->push_back(padding_value);
      for (auto i = 0 ; i < size ; i++) {
        padding_value[i] = paddings.ints(i);
      }
      AddArrayInt32Operand(padding_value, kTensorRank, tensor_dims);
      *nn_op_type = NEURON_PAD;
    } else if (type == "Pooling") {
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      Argument kernels = get_arg(op_def, "kernels");
      AddScalarInt32Operand(kernels.ints(0));  // filter width
      AddScalarInt32Operand(kernels.ints(1));  // filter height
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      Argument pooling_type = get_arg(op_def, "pooling_type");
      if (pooling_type.i() == 1) {
        *nn_op_type = NEURON_AVERAGE_POOL_2D;
      } else if (pooling_type.i() == 2) {
        *nn_op_type = NEURON_MAX_POOL_2D;
      } else {
        LOG(ERROR) << "Unsupport mace pooling mode";
        return false;
      }
    } else if (type == "PRelu") {
      *nn_op_type = NEURON_PRELU;
    } else if (type == "Reduce") {
      Argument reduce_type = get_arg(op_def, "reduce_type");
      Argument axis = get_arg(op_def, "axis");
      Argument keepdims = get_arg(op_def, "keepdims");
      if (reduce_type.i() == 0) {
        *nn_op_type = NEURON_MEAN;
      } else {
        LOG(ERROR) << "Unsupport mace reduce mode";
        return false;
      }
      uint32_t axis_rank = static_cast<uint32_t>(axis.ints_size());
      int32_t *axis_value = nullptr;
      if (axis_rank > 0) {
        axis_value = new int32_t[axis_rank];
        for (auto i = 0 ; i < axis.ints_size() ; i++) {
          axis_value[i] = axis.ints(i);
        }
      }
      int32_buffers_->push_back(axis_value);
      AddVectorInt32Operand(axis_value, axis_rank);
      AddScalarInt32Operand(keepdims.i());
    } else if (type == "Reshape") {
      uint32_t output_shape_rank =
          static_cast<uint32_t>(op_def.output_shape(0).dims_size());
      int32_t *output_shape_value = nullptr;
      if (output_shape_rank > 0) {
        output_shape_value = new int32_t[output_shape_rank];
        for (auto i = 0 ; i < op_def.output_shape(0).dims_size() ; i++) {
          output_shape_value[i] = op_def.output_shape(0).dims(i);
        }
      }
      int32_buffers_->push_back(output_shape_value);
      AddVectorInt32Operand(output_shape_value, output_shape_rank);
      *nn_op_type = NEURON_RESHAPE;
    } else if (type == "ResizeBilinear") {
      Argument align_corners = get_arg(op_def, "align_corners");
      AddScalarInt32Operand(op_def.output_shape(0).dims(1));  // output width
      AddScalarInt32Operand(op_def.output_shape(0).dims(2));  // output height
      // set to true to specify NCHW data layout for input0 and output0.
      AddScalarBoolOperand(false);
      // align_corners
      AddScalarBoolOperand(static_cast<bool>(align_corners.i()));
      *nn_op_type = NEURON_RESIZE_BILINEAR;
    } else if (type == "Softmax") {
      AddScalarFloat32Operand(1.0);
      *nn_op_type = NEURON_SOFTMAX;
    } else if (type == "Deconv2D") {
      // Since bias is the fourth input tensor, we add it here.
      int node_id = op_def.node_input(3).node_id();
      int neuron_tensor_index = operand_mapping_.mace_index_to_neuron(node_id);
      if (neuron_tensor_index == -1) {
        // Allocate a new tensor index
        neuron_tensor_index =
            operand_mapping_.add_new_neuron_tensor_index(node_id);
      }
      augmented_inputs_.push_back(neuron_tensor_index);
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      AddScalarBoolOperand(false);  // Use NHWC format
      *nn_op_type = NEURON_TRANSPOSE_CONV_2D;
    } else if (type == "SpaceToDepth") {
      Argument block_size = get_arg(op_def, "block_size");
      AddScalarInt32Operand(block_size.i());  // block_size
      *nn_op_type = NEURON_SPACE_TO_DEPTH;
    } else if (type == "DepthToSpace") {
      Argument block_size = get_arg(op_def, "block_size");
      AddScalarInt32Operand(block_size.i());  // block_size
      *nn_op_type = NEURON_DEPTH_TO_SPACE;
    } else {
      LOG(ERROR) << "Unsupport mace operator";
      return false;
    }
    return true;
  }

  int HandleFuseCode(OperatorDef op_def, int act_mode) {
      bool is_uint8_type = (op_def.output_type(0) == DT_UINT8);
      float scale = is_uint8_type ? op_def.quantize_info(0).scale() : 0.0;
      int zero_point =
          is_uint8_type ? op_def.quantize_info(0).zero_point() : 0;
      if (act_mode == NEURON_FUSED_RELU6) {
          if (is_uint8_type && zero_point == 0 && 255*scale <= 6.0) {
              return NEURON_FUSED_NONE;
          }
      } else if (act_mode == NEURON_FUSED_RELU) {
          if (is_uint8_type && zero_point == 0) {
              return NEURON_FUSED_NONE;
          }
      }
      return act_mode;
  }

  Argument get_arg(OperatorDef op_def, std::string arg_name) {
      for (auto arg : op_def.arg()) {
        if (arg.name() == arg_name) {
          return arg;
        }
      }
      Argument none;
      return none;
  }

  int GetByteNum(DataType data_type) {
    int byte_per_element;
    if (data_type == DT_FLOAT || data_type == DT_INT32) {
        byte_per_element = 4;
    } else if (data_type == DT_HALF || data_type == DT_INT16) {
        byte_per_element = 2;
    } else if (data_type == DT_UINT8) {
        byte_per_element = 1;
    } else {
      byte_per_element = -1;
      LOG(ERROR) << "unsupport data type";
    }
    return byte_per_element;
  }
};

}  // namespace mace

#endif  // MACE_RUNTIMES_APU_V4_NEURON_DELEGATE_BUILDER_H_
