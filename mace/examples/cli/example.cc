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

#include <malloc.h>
#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
// if convert model to code.
#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

namespace mace {
namespace examples {

namespace str_util {

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result.push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}

}  // namespace str_util

void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!isalnum(res[i])) res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("GPU") == 0) {
    return DeviceType::GPU;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else {
    return DeviceType::CPU;
  }
}


DEFINE_string(model_name,
              "",
              "model name in model deployment file");
DEFINE_string(input_node,
              "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "1,224,224,3:1,1,1,10",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "1,224,224,2:1,1,1,10",
              "output shapes, separated by colon and comma");
DEFINE_string(input_file,
              "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(opencl_binary_file,
              "",
              "compiled opencl binary file path");
DEFINE_string(opencl_parameter_file,
              "",
              "tuned OpenCL parameter file path");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when model_data_format == file");
DEFINE_string(model_file,
              "",
              "model file name, used when load mace model in pb");
DEFINE_string(device, "GPU", "CPU/GPU/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");
#ifndef MODEL_GRAPH_FORMAT_CODE
namespace {
bool ReadBinaryFile(std::vector<unsigned char> *data,
                           const std::string &filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  ifs.seekg(0, ifs.end);
  size_t length = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  data->reserve(length);
  data->insert(data->begin(), std::istreambuf_iterator<char>(ifs),
               std::istreambuf_iterator<char>());
  if (ifs.fail()) {
    return false;
  }
  ifs.close();

  return true;
}
}  // namespace
#endif

bool RunModel(const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes) {
  // load model
  DeviceType device_type = ParseDeviceType(FLAGS_device);
  // config runtime
  mace::SetOpenMPThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
#ifdef MACE_ENABLE_OPENCL
  if (device_type == DeviceType::GPU) {
    mace::SetGPUHints(
        static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
        static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));

    // Just call once. (Not thread-safe)
    // Set paths of Generated OpenCL Compiled Kernel Binary file
    // if you build gpu library of specific soc.
    // Using OpenCL binary will speed up the initialization.
    // OpenCL binary is corresponding to the OpenCL Driver version,
    // you should update the binary when OpenCL Driver changed.
    std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};
    mace::SetOpenCLBinaryPaths(opencl_binary_paths);

    mace::SetOpenCLParameterPath(FLAGS_opencl_parameter_file);
  }
#endif  // MACE_ENABLE_OPENCL

  // DO NOT USE tmp directory.
  // Please use APP's own directory and make sure the directory exists.
  // Just call once
  const std::string internal_storage_path =
      "/data/local/tmp/mace_run/interior";

  // Config internal kv storage factory.
  std::shared_ptr<KVStorageFactory> storage_factory(
      new FileStorageFactory(internal_storage_path));
  SetKVStorageFactory(storage_factory);

  // Create Engine
  std::shared_ptr<mace::MaceEngine> engine;
  MaceStatus create_engine_status;
  // Only choose one of the two type based on the `model_graph_format`
  // in model deployment file(.yml).
#ifdef MODEL_GRAPH_FORMAT_CODE
  // if model_data_format == code, just pass an empty string("")
  // to model_data_file parameter.
  create_engine_status =
      CreateMaceEngineFromCode(FLAGS_model_name,
                               FLAGS_model_data_file,
                               input_names,
                               output_names,
                               device_type,
                               &engine);
#else
  std::vector<unsigned char> model_pb_data;
  if (!ReadBinaryFile(&model_pb_data, FLAGS_model_file)) {
    std::cerr << "Failed to read file: " << FLAGS_model_file << std::endl;
  }
  create_engine_status =
      CreateMaceEngineFromProto(model_pb_data,
                                FLAGS_model_data_file,
                                input_names,
                                output_names,
                                device_type,
                                &engine);
#endif

  if (create_engine_status != MaceStatus::MACE_SUCCESS) {
    std::cerr << "Create engine error, please check the arguments first, "
              << "if correct, the device may not run the model, "
              << "please fall back to other strategy."
              << std::endl;
    exit(1);
  }

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    // load input
    std::ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]),
                          std::ios::in | std::ios::binary);
    if (in_file.is_open()) {
      in_file.read(reinterpret_cast<char *>(buffer_in.get()),
                   input_size * sizeof(float));
      in_file.close();
    } else {
      std::cout << "Open input file failed" << std::endl;
      return -1;
    }
    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
  }

  for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
  }

  std::cout << "Warm up run" << std::endl;
  engine->Run(inputs, &outputs);

  if (FLAGS_round > 0) {
    std::cout << "Run model" << std::endl;
    for (int i = 0; i < FLAGS_round; ++i) {
      engine->Run(inputs, &outputs);
    }
  }

  std::cout << "Write output" << std::endl;
  for (size_t i = 0; i < output_count; ++i) {
    std::string output_name =
        FLAGS_output_file + "_" + FormatName(output_names[i]);
    std::ofstream out_file(output_name, std::ios::binary);
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    out_file.write(
        reinterpret_cast<char *>(outputs[output_names[i]].data().get()),
        output_size * sizeof(float));
    out_file.flush();
    out_file.close();
  }
  std::cout << "Finished" << std::endl;

  return true;
}

int Main(int argc, char **argv) {
  std::string usage = "example run\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "mace version: " << MaceVersion() << std::endl;
  std::cout << "input node: " << FLAGS_input_node << std::endl;
  std::cout << "input shape: " << FLAGS_input_shape << std::endl;
  std::cout << "output node: " << FLAGS_output_node << std::endl;
  std::cout << "output shape: " << FLAGS_output_shape << std::endl;
  std::cout << "input_file: " << FLAGS_input_file << std::endl;
  std::cout << "output_file: " << FLAGS_output_file << std::endl;
  std::cout << "model_data_file: " << FLAGS_model_data_file << std::endl;
  std::cout << "model_file: " << FLAGS_model_file << std::endl;
  std::cout << "device: " << FLAGS_device << std::endl;
  std::cout << "round: " << FLAGS_round << std::endl;
  std::cout << "restart_round: " << FLAGS_restart_round << std::endl;
  std::cout << "gpu_perf_hint: " << FLAGS_gpu_perf_hint << std::endl;
  std::cout << "gpu_priority_hint: " << FLAGS_gpu_priority_hint << std::endl;
  std::cout << "omp_num_threads: " << FLAGS_omp_num_threads << std::endl;
  std::cout << "cpu_affinity_policy: "
            << FLAGS_cpu_affinity_policy
            << std::endl;

  std::vector<std::string> input_names = str_util::Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names =
      str_util::Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes =
      str_util::Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes =
      str_util::Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int64_t>> input_shape_vec(input_count);
  std::vector<std::vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }

  bool ret = false;
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    std::cout << "restart round " << i << std::endl;
    ret =
        RunModel(input_names, input_shape_vec, output_names, output_shape_vec);
  }
  if (ret) {
    return 0;
  } else {
    return -1;
  }
}

}  // namespace examples
}  // namespace mace

int main(int argc, char **argv) { mace::examples::Main(argc, argv); }
