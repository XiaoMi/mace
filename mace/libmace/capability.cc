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

#include <map>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/public/mace.h"

namespace mace {
namespace capability {

const Capability kCPUDefaultCapability = {
    PerformanceInfo {1.f},  // float32_performance
    PerformanceInfo {.6f},  // quantized8_performance tested with mobilenet-v2
    true,  // supported
};

const Capability kGPUDefaultCapability = {
    PerformanceInfo {1.f},  // float32_performance
    PerformanceInfo {-1.f},  // quantized8_performance
    false,  // supported
};

const Capability kDSPDefaultCapability = {
    PerformanceInfo {-1.f},  // float32_performance
    PerformanceInfo {1.f},  // quantized8_performance
    false,  // supported
};

class BMNet {
 public:
  static BMNet *Get();
  MaceStatus Run(DeviceType device,
                 float *exec_time);

 private:
  BMNet();
  void SetUp();

  std::string AddExpandedConv(
      const std::string &blk_name,
      const std::string &input_name,
      const std::vector<int> &dw_strides,
      int input_channel,
      int output_channel,
      const std::vector<std::vector<int>> &feature_map_shapes,
      bool has_expand = false,
      bool has_residual = false);
  void AddConv(
      const std::string &conv_type,
      const std::string &op_name,
      const std::string &input_name,
      const std::string &filter_name,
      const std::string &output_name,
      const std::vector<int> &strides,
      const std::vector<int64_t> &filter_shape,
      const std::vector<int64_t> &output_shape,
      bool has_relu6 = true,
      bool has_bias = true,
      int padding_type = 1);
  void AddEltwise(const std::string &op_name,
                  const std::vector<std::string> &inputs,
                  const std::string &output,
                  const std::vector<int64_t> &output_shape,
                  int type = 0);

  void AddIntArg(OperatorDef *op_def, const std::string &name, int value);
  void AddIntsArg(OperatorDef *op_def,
                  const std::string &name,
                  const std::vector<int> &values);
  void AddStringArg(OperatorDef *op_def,
                    const std::string &name,
                    const std::string &value);
  void AddFloatArg(OperatorDef *op_def, const std::string &name, float value);
  void AddTensor(const std::string &name,
                 const std::vector<int64_t> &shape,
                 int64_t offset,
                 int64_t data_size);

 private:
  // the size of model's weight, used as offset when set up model
  int64_t weight_size_;
  NetDef net_;
  std::vector<unsigned char> weight_;
  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::mutex run_mutex_;
};

BMNet* BMNet::Get() {
  static BMNet net;
  return &net;
}

BMNet::BMNet() : weight_size_(0) {
  SetUp();
}

void BMNet::SetUp() {
  // Use subnet of mobilenet-v2 as the workload
  input_names_.push_back("Input");
  input_shapes_.push_back({1, 224, 224, 3});
  std::string blk_name = "Conv";
  std::string op_output_name = blk_name + "/Conv2D";
  AddConv("Conv2D", blk_name + "/Conv2D",
          "Input", blk_name + "/filter", op_output_name,
          {2, 2}, {32, 3, 3, 3}, {1, 112, 112, 32});
  op_output_name = AddExpandedConv("ExpandedConv", op_output_name, {1, 1},
      32, 16, {{112, 112}, {112, 112}});
  op_output_name = AddExpandedConv("ExpandedConv1", op_output_name, {2, 2},
      16, 24, {{112, 112}, {56, 56}, {56, 56}}, true);
  op_output_name = AddExpandedConv("ExpandedConv2", op_output_name, {1, 1},
      24, 24, {{56, 56}, {56, 56}, {56, 56}}, true, true);
  op_output_name = AddExpandedConv("ExpandedConv3", op_output_name, {2, 2},
      24, 32, {{56, 56}, {28, 28}, {28, 28}}, true);
  op_output_name = AddExpandedConv("ExpandedConv4", op_output_name, {1, 1},
      32, 32, {{28, 28}, {28, 28}, {28, 28}}, true, true);
  op_output_name = AddExpandedConv("ExpandedConv5", op_output_name, {1, 1},
      32, 32, {{28, 28}, {28, 28}, {28, 28}}, true, true);
  op_output_name = AddExpandedConv("ExpandedConv6", op_output_name, {2, 2},
      32, 64, {{28, 28}, {14, 14}, {14, 14}}, true);
  op_output_name = AddExpandedConv("ExpandedConv7", op_output_name, {1, 1},
      64, 64, {{14, 14}, {14, 14}, {14, 14}}, true, true);
  output_names_.push_back(op_output_name);
  output_shapes_.push_back({1, 14, 14, 64});

  // Add input and output information
  for (size_t i = 0; i < input_names_.size(); ++i) {
    InputOutputInfo *info = net_.add_input_info();
    info->set_data_format(static_cast<int>(DataFormat::NHWC));
    info->set_name(input_names_[i]);
    for (auto d : input_shapes_[i]) {
      info->add_dims(static_cast<int>(d));
    }
  }
  for (auto output_name : output_names_) {
    InputOutputInfo *info = net_.add_output_info();
    info->set_name(output_name);
  }
  // allocate weight data
  weight_.resize(weight_size_, 0);
}

void BMNet::AddIntArg(mace::OperatorDef *op_def,
                      const std::string &name,
                      int value) {
  auto arg = op_def->add_arg();
  arg->set_name(name);
  arg->set_i(value);
}

void BMNet::AddIntsArg(mace::OperatorDef *op_def,
                       const std::string &name,
                       const std::vector<int> &values) {
  auto arg = op_def->add_arg();
  arg->set_name(name);
  for (auto value : values) {
    arg->add_ints(value);
  }
}

void BMNet::AddStringArg(mace::OperatorDef *op_def,
                         const std::string &name,
                         const std::string &value) {
  auto arg = op_def->add_arg();
  arg->set_name(name);
  arg->set_s(value);
}

void BMNet::AddFloatArg(mace::OperatorDef *op_def,
                        const std::string &name,
                        float value) {
  auto arg = op_def->add_arg();
  arg->set_name(name);
  arg->set_f(value);
}

void BMNet::AddTensor(const std::string &name,
                      const std::vector<int64_t> &shape,
                      int64_t offset,
                      int64_t data_size) {
  ConstTensor *tensor_ptr = net_.add_tensors();
  tensor_ptr->set_name(name);
  tensor_ptr->mutable_dims()->Reserve(shape.size());
  for (auto dim : shape) {
    tensor_ptr->add_dims(dim);
  }
  tensor_ptr->set_offset(offset);
  tensor_ptr->set_data_size(data_size);
  tensor_ptr->set_data_type(DT_HALF);
}

void BMNet::AddConv(const std::string &conv_type,
                    const std::string &op_name,
                    const std::string &input_name,
                    const std::string &filter_name,
                    const std::string &output_name,
                    const std::vector<int> &strides,
                    const std::vector<int64_t> &filter_shape,
                    const std::vector<int64_t> &output_shape,
                    bool has_relu6,
                    bool has_bias,
                    int padding_type) {
  const int kAlignPad = 4;
  auto op_def = net_.add_op();

  op_def->set_name(op_name);
  op_def->set_type(conv_type);
  op_def->add_input(input_name);
  op_def->add_input(filter_name);
  int64_t filter_size = std::accumulate(filter_shape.begin(),
      filter_shape.end(), 1, std::multiplies<int64_t>());
  AddTensor(filter_name, filter_shape, weight_size_, filter_size);
  weight_size_ += filter_size * sizeof(half);
  if (weight_size_ % kAlignPad != 0) {
    weight_size_ += kAlignPad - (weight_size_ % kAlignPad);
  }
  if (has_bias) {
    std::string bias_tensor_name = op_name + "_bias:0";
    op_def->add_input(bias_tensor_name);
    AddTensor(bias_tensor_name, {output_shape[3]},
              weight_size_, output_shape[3]);
    weight_size_ += output_shape[3] * sizeof(half);
    if (weight_size_ % kAlignPad != 0) {
      weight_size_ += kAlignPad - (weight_size_ % kAlignPad);
    }
  }
  op_def->add_output(output_name);
  AddIntsArg(op_def, "strides", strides);
  AddIntArg(op_def, "padding", padding_type);
  AddIntArg(op_def, "data_format", static_cast<int>(DataFormat::AUTO));
  AddIntArg(op_def, "T", DT_HALF);
  if (has_relu6) {
    AddStringArg(op_def, "activation", "RELUX");
    AddFloatArg(op_def, "max_limit", 6);
  }
  OutputShape *shape = op_def->add_output_shape();
  for (auto dim : output_shape) {
    shape->add_dims(dim);
  }
}

void BMNet::AddEltwise(const std::string &op_name,
                       const std::vector<std::string> &inputs,
                       const std::string &output,
                       const std::vector<int64_t> &output_shape,
                       int type) {
  auto op_def = net_.add_op();

  op_def->set_name(op_name);
  op_def->set_type("Eltwise");
  for (auto input : inputs) {
    op_def->add_input(input);
  }
  op_def->add_output(output);
  AddIntArg(op_def, "type", type);
  AddIntArg(op_def, "T", DT_HALF);
  AddIntArg(op_def, "data_format", static_cast<int>(DataFormat::AUTO));
  OutputShape *shape = op_def->add_output_shape();
  for (auto dim : output_shape) {
    shape->add_dims(dim);
  }
}

std::string BMNet::AddExpandedConv(
    const std::string &blk_name,
    const std::string &input_name,
    const std::vector<int> &dw_strides,
    int input_channel,
    int output_channel,
    const std::vector<std::vector<int>> &feature_map_shapes,
    bool has_expand,
    bool has_residual) {
  int expand_scale = 6;
  std::string middle_input_name = input_name;
  std::string middle_output_name;
  std::string filter_name;
  int expand_out_channel = input_channel;
  int feature_map_idx = 0;
  if (has_expand) {
    std::string expand_conv2d_name = blk_name + "/expand/Conv2D";
    filter_name = expand_conv2d_name + "/filter";
    middle_output_name = expand_conv2d_name + ":0";
    expand_out_channel = input_channel * expand_scale;
    AddConv("Conv2D", expand_conv2d_name, middle_input_name, filter_name,
            middle_output_name, {1, 1},
            {expand_out_channel, input_channel, 1, 1},
            {1, feature_map_shapes[feature_map_idx][0],
             feature_map_shapes[feature_map_idx][1], expand_out_channel});
    feature_map_idx += 1;
    middle_input_name = middle_output_name;
  }

  std::string dw_conv2d_name = blk_name + "/depthwise/depthwise";
  filter_name = dw_conv2d_name + "/filter";
  middle_output_name = dw_conv2d_name + ":0";
  AddConv("DepthwiseConv2d", dw_conv2d_name, middle_input_name, filter_name,
          middle_output_name, dw_strides, {1, expand_out_channel, 3, 3},
          {1, feature_map_shapes[feature_map_idx][0],
           feature_map_shapes[feature_map_idx][1], expand_out_channel});
  feature_map_idx += 1;
  middle_input_name = middle_output_name;

  std::string project_conv2d_name = blk_name + "/project/conv2d";
  filter_name = project_conv2d_name + "/filter";
  middle_output_name = project_conv2d_name + ":0";
  AddConv("Conv2D", project_conv2d_name, middle_input_name, filter_name,
          middle_output_name, {1, 1},
          {output_channel, expand_out_channel, 1, 1},
          {1, feature_map_shapes[feature_map_idx][0],
           feature_map_shapes[feature_map_idx][1], output_channel},
          false);
  middle_input_name = middle_output_name;
  if (has_residual) {
    std::string eltwise_name = blk_name + "/add";
    middle_output_name = eltwise_name + ":0";
    AddEltwise(eltwise_name, {input_name, middle_input_name},
               middle_output_name,
               {1, feature_map_shapes[2][0], feature_map_shapes[2][1],
                output_channel});
  }

  return middle_output_name;
}

MaceStatus BMNet::Run(DeviceType device,
                      float *exec_time) {
  std::lock_guard<std::mutex> lock(run_mutex_);
  MaceStatus status;
  MaceEngineConfig config(device);
  config.SetCPUThreadPolicy(-1, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  if (device == DeviceType::GPU) {
    config.SetGPUHints(GPUPerfHint::PERF_HIGH, GPUPriorityHint::PRIORITY_LOW);
  }
  MaceEngine engine(config);

  status = engine.Init(&net_, input_names_, output_names_, weight_.data());
  if (status != MaceStatus::MACE_SUCCESS) {
    return status;
  }

  const size_t input_count = input_names_.size();
  const size_t output_count = output_names_.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  for (size_t i = 0; i < input_count; ++i) {
    int64_t input_size =
        std::accumulate(input_shapes_[i].begin(), input_shapes_[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    inputs[input_names_[i]] = mace::MaceTensor(input_shapes_[i], buffer_in);
  }

  for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shapes_[i].begin(), output_shapes_[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs[output_names_[i]] = mace::MaceTensor(output_shapes_[i], buffer_out);
  }

  // warm up
  status = engine.Run(inputs, &outputs);
  if (status != MaceStatus::MACE_SUCCESS) {
    return status;
  }

  int round = 10;
  int64_t duration = 0;
  for (int i = 0; i < round; ++i) {
    int64_t start = NowMicros();
    status = engine.Run(inputs, &outputs);
    duration += NowMicros() - start;
    if (status != MaceStatus::MACE_SUCCESS) {
      return status;
    }
  }
  *exec_time = duration / 1000.0f / round;
  return MaceStatus::MACE_SUCCESS;
}
}  // namespace capability

Capability GetCapability(DeviceType device_type, float cpu_float32_exec_time) {
  Capability capability;
  if (device_type == DeviceType::HEXAGON) {
    return capability::kDSPDefaultCapability;
  } else if (device_type == DeviceType::CPU) {
    capability = capability::kCPUDefaultCapability;
  } else if (device_type == DeviceType::GPU) {
    capability = capability::kGPUDefaultCapability;
  } else {
    LOG(FATAL) << "No support the device " << device_type;
  }
  capability::BMNet *net = capability::BMNet::Get();
  float exec_time;
  MaceStatus status = net->Run(device_type, &exec_time);
  if (status == MaceStatus::MACE_SUCCESS) {
    capability.float32_performance.exec_time =
        exec_time / cpu_float32_exec_time;
    capability.supported = true;
  } else {
    capability.supported = false;
  }
  return capability;
}
}  // namespace mace
