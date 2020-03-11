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

#include "mace/core/runtime/apu/apu_wrapper.h"

#include <algorithm>

#include "mace/core/quantize.h"

namespace mace {

ApuWrapper::ApuWrapper(Device *device)
    : quantize_util_(&device->cpu_runtime()->thread_pool()) {
}

apu_data_type ApuWrapper::MapToApuDataType(DataType mace_type) {
  switch (mace_type) {
    case DT_FLOAT:
      return APU_DATA_TYPE_FLOAT;
    case DT_INT32:
      return APU_DATA_TYPE_INT32;
    case DT_HALF:
      return APU_DATA_TYPE_HALF;
    case DT_FLOAT16:
      return APU_DATA_TYPE_HALF;
    case DT_UINT8:
      return APU_DATA_TYPE_UINT8;
    case DT_INT16:
      return APU_DATA_TYPE_INT16;
    default:
      MACE_CHECK(false, "unsupport mace data type");
      break;
  }
  return APU_DATA_TYPE_UNDEFINED;
}

apu_pooling_mode ApuWrapper::MapToApuPoolingMode(int mace_mode) {
  switch (mace_mode) {
    case 1:
      return APU_POOLING_AVG;
    case 2:
      return APU_POOLING_MAX;
    default:
      MACE_CHECK(false, "unsupport mace pooling mode");
      break;
  }
  return APU_POOLING_UNDEFINED;
}

apu_eltwise_mode ApuWrapper::MapToApuEltwiseMode(int mace_mode) {
  switch (mace_mode) {
    case 0:
      return APU_ELTWISE_ADD;
    case 1:
      return APU_ELTWISE_SUB;
    case 2:
      return APU_ELTWISE_MUL;
    case 4:
      return APU_ELTWISE_MIN;
    case 5:
      return APU_ELTWISE_MAX;
    default:
      MACE_CHECK(false, "unsupport mace eltwise mode");
      break;
  }
  return APU_ELTWISE_UNDEFINED;
}

bool ApuWrapper::Init(const NetDef &net_def, unsigned const char *model_data,
                      const char *file_name, bool load, bool store) {
  frontend = new ApuFrontend();

  MACE_CHECK(!(load & store),
            "Should not load and store the model simultaneously.");

  // parse model argument
  int const_data_num = 0;
  int apu_data_type = -1;
  for (auto arg : net_def.arg()) {
    if (arg.name().compare("const_data_num") == 0) {
      const_data_num = arg.i();
    } else if (arg.name().compare("apu_data_type") == 0) {
      apu_data_type = arg.i();
    }
  }
  // input tensors
  std::vector<apu_tensor> input_tensors;
  for (auto input_info : net_def.input_info()) {
    apu_tensor tensor;
    tensor.tensor_id = input_info.node_id();
    tensor.tensor_type = APU_TENSOR_MODEL_INPUT;
    tensor.data_type = MapToApuDataType(static_cast<DataType>(apu_data_type));
    tensor.scale = input_info.has_scale() ? input_info.scale() : -1.0f;
    tensor.zero_point = input_info.has_zero_point() ?
                            input_info.zero_point() : 0;
    tensor.dim_size = input_info.dims_size();
    MACE_CHECK(tensor.dim_size <= APU_TENSOR_MAX_DIMS,
               "tensor dimension size not supported");
    tensor_info info;
    info.name = input_info.name();
    info.size = 1;
    info.data_type = tensor.data_type;
    int byte_per_element = GetByteNum(tensor.data_type);
    for (auto i = 0 ; i < tensor.dim_size ; i++) {
      tensor.dims[i] = input_info.dims(i);
      info.size *= input_info.dims(i);
      info.shape.push_back(input_info.dims(i));
    }
    info.buf
    = std::shared_ptr<uint8_t>(new uint8_t[info.size * byte_per_element],
                               std::default_delete<uint8_t[]>());
    info.scale = tensor.scale;
    info.zero_point = tensor.zero_point;
    input_infos.push_back(info);
    tensor.data_buf = info.buf.get();
    input_tensors.push_back(tensor);
  }
  // output tensors
  std::vector<apu_tensor> output_tensors;
  for (auto output_info : net_def.output_info()) {
    apu_tensor tensor;
    tensor.tensor_id = output_info.node_id();
    tensor.tensor_type = APU_TENSOR_MODEL_OUTPUT;
    tensor.data_type = MapToApuDataType(static_cast<DataType>(apu_data_type));
    tensor.dim_size = output_info.dims_size();
    tensor_info info;
    info.name = output_info.name();
    info.size = 1;
    info.data_type = tensor.data_type;
    int byte_per_element = GetByteNum(tensor.data_type);
    for (auto i = 0 ; i < tensor.dim_size ; i++) {
      tensor.dims[i] = output_info.dims(i);
      info.size *= output_info.dims(i);
      info.shape.push_back(output_info.dims(i));
    }
    info.buf =
    std::shared_ptr<uint8_t>(new uint8_t[info.size * byte_per_element],
                             std::default_delete<uint8_t[]>());
    for (auto op_def : net_def.op()) {
      if (output_info.name() == op_def.output(0)) {
        if (info.data_type == static_cast<int>(APU_DATA_TYPE_UINT8) ||
            info.data_type == static_cast<int>(APU_DATA_TYPE_INT16)) {
          info.scale = op_def.quantize_info(0).scale();
          info.zero_point = op_def.quantize_info(0).zero_point();
        } else {
          info.scale = 0.0;
          info.zero_point = 0;
        }
      }
    }
    output_infos.push_back(info);
    tensor.data_buf = info.buf.get();
    output_tensors.push_back(tensor);
  }
  // const tensors
  std::vector<apu_tensor> const_tensors;
  // operators
  std::vector<apu_operator> ops;
  std::vector<std::vector<int>> cached_op_inputs;
  if (!load) {
    // const tensors
    for (auto const_tensor : net_def.tensors()) {
      apu_tensor tensor;
      tensor.tensor_id = const_tensor.node_id();
      tensor.tensor_type = (tensor.tensor_id < const_data_num) ?
                               APU_TENSOR_CONST_DATA :
                               APU_TENSOR_CONST_ARGUMENT;
      tensor.data_type = MapToApuDataType(const_tensor.data_type());
      tensor.scale = const_tensor.has_scale() ? const_tensor.scale() : 0.0f;
      tensor.zero_point = const_tensor.has_zero_point() ?
                              const_tensor.zero_point() : 0;
      tensor.dim_size = const_tensor.dims_size();
      MACE_CHECK(tensor.dim_size <= APU_TENSOR_MAX_DIMS,
                 "tensor dimension size not supported");
      for (auto i = 0 ; i < tensor.dim_size ; i++) {
        tensor.dims[i] = const_tensor.dims(i);
      }
      tensor.data_buf =
          const_cast<unsigned char*>(model_data + const_tensor.offset());
      const_tensors.push_back(tensor);
    }
    // operators
    for (auto op_def : net_def.op()) {
      apu_operator op;
      strncpy(op.type, op_def.type().c_str(), APU_OP_TYPE_MAX_SIZE);
      op.input_size = op_def.node_input_size();
      std::vector<int> input_ids;
      for (auto i = 0 ; i < op.input_size ; i++) {
        input_ids.push_back(op_def.node_input(i).node_id());
      }
      cached_op_inputs.push_back(input_ids);
      op.input_ids = cached_op_inputs.back().data();
      op.output.tensor_id = op_def.node_id();
      op.output.tensor_type = APU_TENSOR_OP_OUTPUT;
      op.output.data_type = MapToApuDataType(op_def.output_type(0));
      if (op.output.data_type == APU_DATA_TYPE_UINT8 ||
          op.output.data_type == APU_DATA_TYPE_INT16) {
        op.output.scale = op_def.quantize_info(0).scale();
        op.output.zero_point = op_def.quantize_info(0).zero_point();
      } else {
        op.output.scale = 0.0f;
        op.output.zero_point = 0;
      }
      op.output.dim_size = op_def.output_shape(0).dims_size();
      MACE_CHECK(op.output.dim_size <= APU_TENSOR_MAX_DIMS,
                 "tensor dimension size not supported");
      for (auto i = 0 ; i < op.output.dim_size ; i++) {
        op.output.dims[i] = op_def.output_shape(0).dims(i);
      }
      op.output.data_buf = nullptr;
      // get op mode and activation mode
      bool is_pooling = (strcmp(op.type, "Pooling") == 0);
      bool is_eltwise = (strcmp(op.type, "Eltwise") == 0);
      std::string activation;
      float max_limit = 0.0f;
      for (auto arg : op_def.arg()) {
        if (arg.name().compare("activation") == 0) {
          activation = arg.s();
        }
        if (arg.name().compare("max_limit") == 0) {
          max_limit = arg.f();
        }
        if (is_pooling && arg.name().compare("pooling_type") == 0) {
          op.op_mode = static_cast<int>(MapToApuPoolingMode(arg.i()));
        }
        if (is_eltwise && arg.name().compare("type") == 0) {
          op.op_mode = static_cast<int>(MapToApuEltwiseMode(arg.i()));
        }
      }
      if (activation.compare("RELU") == 0) {
        op.act_mode = APU_ACT_RELU;
      } else if (activation.compare("RELUX") == 0 && max_limit == 6.0) {
        op.act_mode = APU_ACT_RELU6;
      } else if (activation.compare("SIGMOID") == 0) {
        op.act_mode = APU_ACT_SIGMOID;
      } else if (activation.compare("TANH") == 0) {
        op.act_mode = APU_ACT_TANH;
      } else {
        op.act_mode = APU_ACT_NONE;
      }
      ops.push_back(op);
    }
  }
  bool print_model = false;
  bool ret = frontend->InitGraph(
                 const_tensors.size(), const_tensors.data(),
                 input_tensors.size(), input_tensors.data(),
                 output_tensors.size(), output_tensors.data(),
                 ops.size(), ops.data(),
                 print_model, file_name, load, store);
  cached_op_inputs.clear();
  return ret;
}

bool ApuWrapper::Run(const std::map<std::string, Tensor *> &input_tensors,
                     std::map<std::string, Tensor *> *output_tensors) {
  MACE_ASSERT(input_tensors.size() == input_infos.size(), "Wrong inputs num");
  MACE_ASSERT(output_tensors.size() == output_infos.size(),
              "Wrong outputs num");
  // prepare input
  for (int i = 0 ; i < static_cast<int>(input_tensors.size()) ; i++) {
    Tensor* tensor = input_tensors.at(input_infos[i].name);

    // check size
    int element_size = input_infos[i].size;
    int byte_per_element = GetByteNum(input_infos[i].data_type);
    MACE_ASSERT(element_size == static_cast<int>(tensor->size()),
                "Wrong input size");
    // quantize
    if (input_infos[i].data_type == APU_DATA_TYPE_INT16) {
      Quantize16bit(
          (const float*)tensor->raw_data(),
          element_size,
          input_infos[i].scale,
          input_infos[i].zero_point,
          reinterpret_cast<int16_t*>(input_infos[i].buf.get()));
    } else if (input_infos[i].data_type == APU_DATA_TYPE_FLOAT) {
        std::memcpy(input_infos[i].buf.get(),
                    (const float*)tensor->raw_data(),
                    element_size * byte_per_element);
    } else {
      quantize_util_.QuantizeWithScaleAndZeropoint(
          (const float*)tensor->raw_data(),
          element_size,
          input_infos[i].scale,
          input_infos[i].zero_point,
          input_infos[i].buf.get());
    }
  }

  // run model
  bool ret = frontend->RunGraph();
  MACE_CHECK(ret == true, "neuron run model failed");

  // process output
  for (int i = 0 ; i < static_cast<int>(output_tensors->size()) ; i++) {
    Tensor* tensor = output_tensors->at(output_infos[i].name);

    // prepare out buffer
    tensor->SetDtype(DT_FLOAT);
    tensor->Resize(output_infos[i].shape);
    int element_size = output_infos[i].size;
    int byte_per_element = GetByteNum(output_infos[i].data_type);
    MACE_ASSERT(element_size == static_cast<int>(tensor->size()),
                "Wrong output size");
    // dequantize
    if (output_infos[i].data_type == APU_DATA_TYPE_INT16) {
      Dequantize16bit(
          reinterpret_cast<int16_t*>(output_infos[i].buf.get()),
          element_size,
          output_infos[i].scale,
          output_infos[i].zero_point,
          reinterpret_cast<float*>(tensor->raw_mutable_data()));
    } else if (output_infos[i].data_type == APU_DATA_TYPE_FLOAT) {
        std::memcpy(reinterpret_cast<float*>(tensor->raw_mutable_data()),
                    output_infos[i].buf.get(),
                    element_size * byte_per_element);
    } else {
      quantize_util_.Dequantize(
          output_infos[i].buf.get(),
          element_size,
          output_infos[i].scale,
          output_infos[i].zero_point,
          reinterpret_cast<float*>(tensor->raw_mutable_data()));
    }
  }

  return true;
}

bool ApuWrapper::Uninit() {
    bool ret = frontend->UninitGraph();
    frontend = nullptr;
    input_infos.clear();
    output_infos.clear();
    return ret;
}

void ApuWrapper::Quantize16bit(
    const float *input,
    const index_t size,
    float scale,
    int32_t zero_point,
    int16_t *output) {

  const float recip_scale = 1.f / scale;
  for (index_t i = 0; i < size; ++i) {
    output[i] =
    Saturate<int16_t>(roundf(zero_point + recip_scale * input[i]));
  }
}

void ApuWrapper::Dequantize16bit(
    const int16_t *input,
    const index_t size,
    const float scale,
    const int32_t zero_point,
    float *output) {

  for (index_t i = 0; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}

int ApuWrapper::GetByteNum(apu_data_type data_type) {
    int byte_per_element;
    if (data_type == APU_DATA_TYPE_FLOAT || data_type == APU_DATA_TYPE_INT32) {
        byte_per_element = 4;
    } else if (data_type == APU_DATA_TYPE_HALF ||
               data_type == APU_DATA_TYPE_INT16) {
        byte_per_element = 2;
    } else if (data_type == APU_DATA_TYPE_UINT8) {
        byte_per_element = 1;
    } else {
      byte_per_element = 1;
      MACE_CHECK(false, "unsupport data type");
    }
    return byte_per_element;
}

}  // namespace mace
