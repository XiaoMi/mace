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

#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <thread>  // NOLINT(build/c++11)

#include "gflags/gflags.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"
#include "mace/utils/statistics.h"
#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

namespace mace {
namespace benchmark {

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
    if (!::isalnum(res[i])) res[i] = '_';
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

DataFormat ParseDataFormat(const std::string &data_format_str) {
  if (data_format_str == "NHWC") {
    return DataFormat::NHWC;
  } else if (data_format_str == "NCHW") {
    return DataFormat::NCHW;
  } else if (data_format_str == "OIHW") {
    return DataFormat::OIHW;
  } else {
    return DataFormat::NONE;
  }
}

bool RunInference(MaceEngine *engine,
                  const std::map<std::string, mace::MaceTensor> &input_infos,
                  std::map<std::string, mace::MaceTensor> *output_infos,
                  int64_t *inference_time_us,
                  OpStat *statistician) {
  MACE_CHECK_NOTNULL(output_infos);
  RunMetadata run_metadata;
  RunMetadata *run_metadata_ptr = nullptr;
  if (statistician) {
    run_metadata_ptr = &run_metadata;
  }

  const int64_t start_time = NowMicros();
  mace::MaceStatus s = engine->Run(input_infos, output_infos, run_metadata_ptr);
  const int64_t end_time = NowMicros();

  if (s != mace::MaceStatus::MACE_SUCCESS) {
    LOG(ERROR) << "Error during inference.";
    return false;
  }
  *inference_time_us = end_time - start_time;

  if (statistician != nullptr) {
    statistician->StatMetadata(run_metadata);
  }

  return true;
}

bool Run(const std::string &title,
         MaceEngine *engine,
         const std::map<std::string, mace::MaceTensor> &input_infos,
         std::map<std::string, mace::MaceTensor> *output_infos,
         int num_runs,
         double max_time_sec,
         int64_t *total_time_us,
         int64_t *actual_num_runs,
         OpStat *statistician) {
  MACE_CHECK_NOTNULL(output_infos);
  *total_time_us = 0;

  TimeInfo<int64_t> time_info;

  bool util_max_time = (num_runs <= 0);
  for (int i = 0; util_max_time || i < num_runs; ++i) {
    int64_t inference_time_us = 0;
    bool s = RunInference(engine, input_infos, output_infos,
                          &inference_time_us, statistician);
    time_info.UpdateTime(inference_time_us);
    (*total_time_us) += inference_time_us;
    ++(*actual_num_runs);

    if (max_time_sec > 0 && (*total_time_us / 1000000.0) > max_time_sec) {
      break;
    }

    if (!s) {
      LOG(INFO) << "Failed on run " << i;
      return s;
    }
  }

  std::stringstream stream(time_info.ToString(title));
  stream << std::endl;
  for (std::string line; std::getline(stream, line);) {
    LOG(INFO) << line;
  }
  return true;
}

DEFINE_string(model_name, "", "model name in yaml");
DEFINE_string(device, "CPU", "Device [CPU|GPU|DSP]");
DEFINE_string(input_node, "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(output_node, "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(input_shape, "", "input shape, separated by colon and comma");
DEFINE_string(output_shape, "", "output shape, separated by colon and comma");
DEFINE_string(input_data_format,
              "NHWC",
              "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format,
              "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file, "", "input file name");
DEFINE_int32(max_num_runs, 100, "max number of runs");
DEFINE_double(max_seconds, 10.0, "max number of seconds to run");
DEFINE_int32(warmup_runs, 1, "how many runs to initialize model");
DEFINE_string(opencl_binary_file,
              "",
              "compiled opencl binary file path");
DEFINE_string(opencl_parameter_file,
              "",
              "tuned OpenCL parameter file path");
DEFINE_string(model_data_file, "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_string(model_file, "",
              "model file name, used when load mace model in pb");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");

int Main(int argc, char **argv) {
  MACE_CHECK(FLAGS_device != "HEXAGON",
             "Model benchmark tool do not support DSP.");
  std::string usage = "benchmark model\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Model name: [" << FLAGS_model_name << "]";
  LOG(INFO) << "Model_file: " << FLAGS_model_file;
  LOG(INFO) << "Device: [" << FLAGS_device << "]";
  LOG(INFO) << "gpu_perf_hint: [" << FLAGS_gpu_perf_hint << "]";
  LOG(INFO) << "gpu_priority_hint: [" << FLAGS_gpu_priority_hint << "]";
  LOG(INFO) << "omp_num_threads: [" << FLAGS_omp_num_threads << "]";
  LOG(INFO) << "cpu_affinity_policy: [" << FLAGS_cpu_affinity_policy << "]";
  LOG(INFO) << "Input node: [" << FLAGS_input_node<< "]";
  LOG(INFO) << "Input shapes: [" << FLAGS_input_shape << "]";
  LOG(INFO) << "Output node: [" << FLAGS_output_node<< "]";
  LOG(INFO) << "output shapes: [" << FLAGS_output_shape << "]";
  LOG(INFO) << "Warmup runs: [" << FLAGS_warmup_runs << "]";
  LOG(INFO) << "Num runs: [" << FLAGS_max_num_runs << "]";
  LOG(INFO) << "Max run seconds: [" << FLAGS_max_seconds << "]";

  std::unique_ptr<OpStat> statistician(new OpStat());

  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes = Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = Split(FLAGS_output_shape, ':');

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

  std::vector<std::string> raw_input_data_formats =
      Split(FLAGS_input_data_format, ',');
  std::vector<std::string> raw_output_data_formats =
      Split(FLAGS_output_data_format, ',');
  std::vector<DataFormat> input_data_formats(input_count);
  std::vector<DataFormat> output_data_formats(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_data_formats[i] = ParseDataFormat(raw_input_data_formats[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    output_data_formats[i] = ParseDataFormat(raw_output_data_formats[i]);
  }

  mace::DeviceType device_type = ParseDeviceType(FLAGS_device);

  // configuration
  MaceStatus mace_status;
  MaceEngineConfig config(device_type);
  mace_status = config.SetCPUThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
  if (mace_status != MaceStatus::MACE_SUCCESS) {
    LOG(INFO) << "Set openmp or cpu affinity failed.";
  }
#ifdef MACE_ENABLE_OPENCL
  std::shared_ptr<GPUContext> gpu_context;
  if (device_type == DeviceType::GPU) {
    // DO NOT USE tmp directory.
    // Please use APP's own directory and make sure the directory exists.
    const char *storage_path_ptr = getenv("MACE_INTERNAL_STORAGE_PATH");
    const std::string storage_path =
        std::string(storage_path_ptr == nullptr ?
                    "/data/local/tmp/mace_run/interior" : storage_path_ptr);
    std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};

    gpu_context = GPUContextBuilder()
        .SetStoragePath(storage_path)
        .SetOpenCLBinaryPaths(opencl_binary_paths)
        .SetOpenCLParameterPath(FLAGS_opencl_parameter_file)
        .Finalize();

    config.SetGPUContext(gpu_context);
    config.SetGPUHints(
        static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
        static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
  }
#endif  // MACE_ENABLE_OPENCL

  // Create Engine
  std::shared_ptr<mace::MaceEngine> engine;
  MaceStatus create_engine_status;
  // Create Engine
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_file != "") {
    auto fs = GetFileSystem();
    auto status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
        &model_graph_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
    }
  }

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_data_file != "") {
    auto fs = GetFileSystem();
    auto status = fs->NewReadOnlyMemoryRegionFromFile(
        FLAGS_model_data_file.c_str(),
        &model_weights_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
    }
    MACE_CHECK(model_weights_data->length() > 0);
  }

#ifdef MODEL_GRAPH_FORMAT_CODE
  create_engine_status = CreateMaceEngineFromCode(FLAGS_model_name,
      reinterpret_cast<const unsigned char *>(model_weights_data->data()),
      model_weights_data->length(),
      input_names,
      output_names,
      config,
      &engine);
#else
  create_engine_status = CreateMaceEngineFromProto(
      reinterpret_cast<const unsigned char *>(model_graph_data->data()),
      model_graph_data->length(),
      reinterpret_cast<const unsigned char *>(model_weights_data->data()),
      model_weights_data->length(),
      input_names,
      output_names,
      config,
      &engine);
#endif
  if (create_engine_status != MaceStatus::MACE_SUCCESS) {
    LOG(FATAL) << "Create engine error, please check the arguments";
  }

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  for (size_t i = 0; i < input_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t input_size =
        std::accumulate(input_shape_vec[i].begin(), input_shape_vec[i].end(), 4,
                        std::multiplies<int64_t>());
    auto buffer_in = std::shared_ptr<char>(new char[input_size],
                                            std::default_delete<char[]>());
    // load input
    std::ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]),
                          std::ios::in | std::ios::binary);
    if (in_file.is_open()) {
      in_file.read(buffer_in.get(), input_size);
      in_file.close();
    } else {
      LOG(INFO) << "Open input file failed";
      return -1;
    }
    inputs[input_names[i]] = mace::MaceTensor(input_shape_vec[i], buffer_in,
        input_data_formats[i]);
  }

  for (size_t i = 0; i < output_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t output_size =
        std::accumulate(output_shape_vec[i].begin(),
                        output_shape_vec[i].end(), 4,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                            std::default_delete<char[]>());
    outputs[output_names[i]] = mace::MaceTensor(output_shape_vec[i],
                                                buffer_out,
                                                output_data_formats[i]);
  }

  int64_t warmup_time_us = 0;
  int64_t num_warmup_runs = 0;
  if (FLAGS_warmup_runs > 0) {
    bool status =
        Run("Warm Up", engine.get(), inputs, &outputs,
            FLAGS_warmup_runs, -1.0,
            &warmup_time_us, &num_warmup_runs, nullptr);
    if (!status) {
      LOG(ERROR) << "Failed at warm up run";
    }
  }

  int64_t no_stat_time_us = 0;
  int64_t no_stat_runs = 0;
  bool status =
      Run("Run without statistics", engine.get(), inputs, &outputs,
          FLAGS_max_num_runs, FLAGS_max_seconds,
          &no_stat_time_us, &no_stat_runs, nullptr);
  if (!status) {
    LOG(ERROR) << "Failed at normal no-stat run";
  }

  int64_t stat_time_us = 0;
  int64_t stat_runs = 0;
  status = Run("Run with statistics", engine.get(), inputs, &outputs,
               FLAGS_max_num_runs, FLAGS_max_seconds,
               &stat_time_us, &stat_runs, statistician.get());
  if (!status) {
    LOG(ERROR) << "Failed at normal stat run";
  }

  statistician->PrintStat();

  return 0;
}

}  // namespace benchmark
}  // namespace mace

int main(int argc, char **argv) { mace::benchmark::Main(argc, argv); }
