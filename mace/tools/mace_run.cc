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

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --model_data_file=model_data.data
 */
#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/core/runtime/runtime.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/statistics.h"
#include "mace/utils/transpose.h"
#include "mace/utils/dbg.h"

#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

namespace mace {
namespace tools {

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

IDataType ParseDataType(const std::string &data_type_str) {
  if (data_type_str == "float32" || data_type_str == "1") {
    return IDataType::IDT_FLOAT;
  } else if (data_type_str == "float16" || data_type_str == "5") {
    return IDataType::IDT_FLOAT16;
  } else if (data_type_str == "bfloat16" || data_type_str == "6") {
    return IDataType::IDT_BFLOAT16;
  } else if (data_type_str == "int16" || data_type_str == "7") {
    return IDataType::IDT_INT16;
  } else if (data_type_str == "uint8" || data_type_str == "2") {
    return IDataType::IDT_UINT8;
  } else if (data_type_str == "int32" || data_type_str == "4") {
    return IDataType::IDT_INT32;
  } else {
    return IDataType::IDT_FLOAT;
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

DEFINE_string(model_name,
              "",
              "model name in yaml");
DEFINE_string(input_node,
              "",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "",
              "output shapes, separated by colon and comma");
DEFINE_string(input_data_type,
              "float32",
              "input data type, NONE|float32|float16|bfloat16");
DEFINE_string(output_data_type,
              "float32",
              "output data type, NONE|float32|float16|bfloat16");
DEFINE_string(input_data_format,
              "NHWC",
              "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format,
              "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file,
              "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(input_dir,
              "",
              "input directory name");
DEFINE_string(output_dir,
              "output",
              "output directory name");
DEFINE_string(opencl_cache_full_path,
              "",
              "opencl cache file path");
DEFINE_string(opencl_binary_file,
              "",
              "compiled opencl binary file path: "
              "will be deprecated in the future, use opencl_cache_full_path");
DEFINE_string(opencl_parameter_file,
              "",
              "tuned OpenCL parameter file path");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
DEFINE_string(model_file,
              "",
              "model file name, used when load mace model in pb");
DEFINE_string(
    accelerator_binary_file,
    "",
    "accelerator init cache path, used when load accelerator init cache");
DEFINE_string(
    accelerator_storage_file,
    "",
    "accelerator init cache path, used when store accelerator init cache");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(num_threads, -1, "num of threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");
DEFINE_int32(apu_boost_hint, 100,
             "APU boost value ranged between 0 (lowest) to 100 (highest)");
DEFINE_int32(apu_preference_hint, 1,
             "0:NEURON_PREFER_LOW_POWER"
             "1:NEURON_PREFER_FAST_SINGLE_ANSWER"
             "2:NEURON_PREFER_SUSTAINED_SPEED");
DEFINE_int32(opencl_cache_reuse_policy,
            1,
            "0:NONE/1:REUSE_SAME_GPU");
DEFINE_int32(accelerator_cache_policy, 0, "0:NONE/1:STORE/2:LOAD/3:APU_LOAD_OR_STORE");
DEFINE_bool(benchmark, false, "enable benchmark op");
DEFINE_bool(fake_warmup, false, "enable fake warmup");

namespace {
std::shared_ptr<char> ReadInputDataFromFile(
    const std::string &file_path, const int tensor_size,
    const IDataType input_data_type,
    std::shared_ptr<char> input_data = nullptr) {
  auto file_data_size = tensor_size * GetEnumTypeSize(DT_FLOAT);
  auto buffer_in = std::shared_ptr<char>(new char[file_data_size],
                                         std::default_delete<char[]>());
  std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
  if (in_file.is_open()) {
    in_file.read(buffer_in.get(), file_data_size);
    in_file.close();
  } else {
    LOG(FATAL) << "Open input file(" << file_path << ") failed";
    return nullptr;
  }

  auto input_size =
      tensor_size * GetEnumTypeSize(static_cast<DataType>(input_data_type));
  if (input_data == nullptr) {
    input_data = std::shared_ptr<char>(new char[input_size],
                                       std::default_delete<char[]>());
  }
  // CopyDataBetweenSameType and CopyDataBetweenDiffType are not an exported
  // functions, app should not use it, the follow line is only used to
  // transform data from file during testing.
  if (input_data_type == IDT_FLOAT || input_data_type == IDT_INT32) {
    mace::ops::CopyDataBetweenSameType(
        nullptr, buffer_in.get(), input_data.get(), input_size);
#ifdef MACE_ENABLE_FP16
  } else if (input_data_type == IDT_FLOAT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const float *>(buffer_in.get()),
        reinterpret_cast<half *>(input_data.get()), tensor_size);
#endif  // MACE_ENABLE_FP16
#ifdef MACE_ENABLE_BFLOAT16
  } else if (input_data_type == IDT_BFLOAT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const float *>(buffer_in.get()),
        reinterpret_cast<BFloat16 *>(input_data.get()), tensor_size);
#endif  // MACE_ENABLE_BFLOAT16
#ifdef MACE_ENABLE_MTK_APU
  } else if (input_data_type == IDT_INT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const float *>(buffer_in.get()),
        reinterpret_cast<int16_t *>(input_data.get()), tensor_size);
  } else if (input_data_type == IDT_UINT8) {
    LOG(INFO) << "read uint8 data from file";
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const float *>(buffer_in.get()),
        reinterpret_cast<uint8_t *>(input_data.get()), tensor_size);
#endif  // MACE_ENABLE_MTK_APU
  } else {
    LOG(FATAL) << "Input data type " << input_data_type << " is not supported.";
  }

  return input_data;
}

int64_t WriteOutputDataToFile(const std::string &file_path,
                              const IDataType file_data_type,
                              const std::shared_ptr<void> output_data,
                              const IDataType output_data_type,
                              const std::vector<int64_t> &output_shape) {
  int64_t output_size = std::accumulate(output_shape.begin(),
                                        output_shape.end(), 1,
                                        std::multiplies<int64_t>());
  auto output_bytes = output_size * sizeof(float);
  std::vector<float> tmp_output(output_size);
  // CopyDataBetweenSameType and CopyDataBetweenDiffType are not an exported
  // functions, app should not use it, the follow line is only used to
  // transform data from file during testing.
  if (file_data_type == output_data_type) {
    mace::ops::CopyDataBetweenSameType(
        nullptr, output_data.get(), tmp_output.data(), output_bytes);
#ifdef MACE_ENABLE_FP16
  } else if (file_data_type == IDT_FLOAT && output_data_type == IDT_FLOAT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const half *>(output_data.get()),
        reinterpret_cast<float *>(tmp_output.data()), output_size);
#endif  // MACE_ENABLE_FP16
#ifdef MACE_ENABLE_BFLOAT16
  } else if (file_data_type == IDT_FLOAT && output_data_type == IDT_BFLOAT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const BFloat16 *>(output_data.get()),
        reinterpret_cast<float *>(tmp_output.data()), output_size);
#endif  // MACE_ENABLE_BFLOAT16
#ifdef MACE_ENABLE_MTK_APU
  } else if (file_data_type == IDT_FLOAT && output_data_type == IDT_UINT8) {
    LOG(INFO) << "write uint8 data to file";
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const uint8_t *>(output_data.get()),
        reinterpret_cast<float *>(tmp_output.data()), output_size);
  } else if (file_data_type == IDT_FLOAT && output_data_type == IDT_INT16) {
    mace::ops::CopyDataBetweenDiffType(
        nullptr, reinterpret_cast<const int16_t *>(output_data.get()),
        reinterpret_cast<float *>(tmp_output.data()), output_size);
#endif  // MACE_ENABLE_MTK_APU
  } else {
    LOG(FATAL) << "Output data type " << output_data_type <<
               " is not supported.";
  }

  std::ofstream out_file(file_path, std::ios::binary);
  MACE_CHECK(out_file.is_open(), "Open output file failed: ", strerror(errno));
  out_file.write(reinterpret_cast<char *>(tmp_output.data()), output_bytes);
  out_file.flush();
  out_file.close();

  return output_size;
}

void PrintRuntimes(const std::vector<RuntimeType> &runtime_types) {
  static std::unordered_map<int, std::string> runtime_names = {
      {RT_CPU, "CPU"},
      {RT_OPENCL, "GPU"},
      {RT_HEXAGON, "DSP"},
      {RT_HTA, "HTA"},
      {RT_HTP, "HTP"},
      {RT_APU, "APU"},
  };
  std::vector<std::string> runtime_strings(runtime_types.size());
  for (size_t i = 0; i < runtime_types.size(); ++i) {
    runtime_strings[i] = runtime_names[runtime_types[i]];
  }

  LOG(INFO) << "runtimes: " << MakeString(runtime_strings);
}
}  // namespace


bool RunModel(const std::string &model_name,
              const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<IDataType> &input_data_types,
              const std::vector<DataFormat> &input_data_formats,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes,
              const std::vector<IDataType> &output_data_types,
              const std::vector<DataFormat> &output_data_formats,
              float cpu_capability) { 
  int64_t t0 = NowMicros();
  bool *model_data_unused = nullptr;
  MaceEngine *tutor = nullptr;

  MaceStatus status;
  // Graph's runtime is set in the yml file, you can use config.SetRuntimeType
  // To dynamically adjust the runtime type
  MaceEngineConfig config;
  status = config.SetCPUThreadPolicy(
      FLAGS_num_threads,
      static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
  if (status != MaceStatus::MACE_SUCCESS) {
    LOG(WARNING) << "Set cpu affinity failed.";
  }
#if defined(MACE_ENABLE_OPENCL) || defined(MACE_ENABLE_HTA)
  std::shared_ptr<OpenclContext> opencl_context;
  const char *storage_path_ptr = getenv("MACE_INTERNAL_STORAGE_PATH");
  const std::string storage_path =
      std::string(storage_path_ptr == nullptr ?
                  "/data/local/tmp/mace_run/interior" : storage_path_ptr);
  std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};

  opencl_context = GPUContextBuilder()
      .SetStoragePath(storage_path)
      .SetOpenCLCacheFullPath(FLAGS_opencl_cache_full_path)
      .SetOpenCLCacheReusePolicy(
          static_cast<OpenCLCacheReusePolicy>(FLAGS_opencl_cache_reuse_policy))
      .SetOpenCLBinaryPaths(opencl_binary_paths)
      .SetOpenCLParameterPath(FLAGS_opencl_parameter_file)
      .Finalize();

  config.SetGPUContext(opencl_context);
  config.SetGPUHints(
      static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
      static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
#endif  // MACE_ENABLE_OPENCL
#ifdef MACE_ENABLE_HEXAGON
  // SetHexagonToUnsignedPD() can be called for 8150 family(with new cDSP
  // firmware) or 8250 family above to run hexagon nn on unsigned PD.
#ifdef __ANDROID__
  config.SetHexagonToUnsignedPD();
#endif
  config.SetHexagonPower(HEXAGON_NN_CORNER_TURBO, true, 100);
#endif
#ifdef MACE_ENABLE_MTK_APU
  config.SetAPUHints(FLAGS_apu_boost_hint,
                     static_cast<APUPreferenceHint>(FLAGS_apu_preference_hint));
#endif
#if defined(MACE_ENABLE_MTK_APU) || defined(MACE_ENABLE_QNN)
  config.SetAcceleratorCache(
      static_cast<AcceleratorCachePolicy>(FLAGS_accelerator_cache_policy),
      FLAGS_accelerator_binary_file, FLAGS_accelerator_storage_file);
#endif
#ifdef MACE_ENABLE_QNN
  config.SetQnnPerformance(HEXAGON_SYSTEM_SETTINGS);
#endif
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
      make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
                                                 &model_graph_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
    }
  }

  // model_weights_data should be kept the lifetime of MaceEngine if device_type
  // is CPU except half/uint8 weights are used to compress model data size.
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
      make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_data_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
                                                 &model_weights_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
    }
  }

  std::shared_ptr<mace::MaceEngine> engine;
  MaceStatus create_engine_status;

  while (true) {
    // Create Engine
    int64_t t0 = NowMicros();
#ifdef MODEL_GRAPH_FORMAT_CODE
    if (model_name.empty()) {
      LOG(INFO) << "Please specify model name you want to run";
      return false;
    }
    create_engine_status =
          CreateMaceEngineFromCode(model_name,
                                   reinterpret_cast<const unsigned char *>(
                                     model_weights_data->data()),
                                   model_weights_data->length(),
                                   input_names,
                                   output_names,
                                   config,
                                   &engine,
                                   model_data_unused,
                                   tutor,
                                   FLAGS_fake_warmup);
#else
    (void) (model_name);
    if (model_graph_data == nullptr || model_weights_data == nullptr) {
      LOG(INFO) << "Please specify model graph file and model data file";
      return false;
    }
    create_engine_status =
        CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                      model_graph_data->data()),
                                  model_graph_data->length(),
                                  reinterpret_cast<const unsigned char *>(
                                      model_weights_data->data()),
                                  model_weights_data->length(),
                                  input_names,
                                  output_names,
                                  config,
                                  &engine,
                                  model_data_unused,
                                  tutor,
                                  FLAGS_fake_warmup);
#endif
    //std::this_thread::sleep_for(std::chrono::seconds(20));
    int64_t t1 = NowMicros();

    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Create engine runtime error, retry ... errcode: "
                 << create_engine_status.information();
    } else {
      double create_engine_millis = (t1 - t0) / 1000.0;
      LOG(INFO) << "Create Mace Engine latency: " << create_engine_millis
                << " ms";
      break;
    }
  }
  int64_t t1 = NowMicros();
  double init_millis = (t1 - t0) / 1000.0;
  LOG(INFO) << "Total init latency: " << init_millis << " ms";
  PrintRuntimes(engine->GetRuntimeTypes());

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  std::map<std::string, int64_t> inputs_size;
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    // only support float and int32, use char for generalization
    // sizeof(int) == 4, sizeof(float) == 4
    auto input_tensor_size = std::accumulate(
        input_shapes[i].begin(), input_shapes[i].end(), 1,
        std::multiplies<int64_t>());
    auto file_path = FLAGS_input_file + "_" + FormatName(input_names[i]);
    LOG(INFO) << "intput_file: " << file_path;
    auto input_data = ReadInputDataFromFile(
        file_path, input_tensor_size, input_data_types[i]);

    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], input_data,
        input_data_formats[i], input_data_types[i]);
    inputs_size[input_names[i]] = input_tensor_size;
  }

  for (size_t i = 0; i < output_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 4,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                            std::default_delete<char[]>());
    outputs[output_names[i]] = mace::MaceTensor(
        output_shapes[i], buffer_out, output_data_formats[i],
        static_cast<IDataType>(output_data_types[i]));
  }

  if (!FLAGS_input_dir.empty()) {
    DIR *dir_parent;
    struct dirent *entry;
    dir_parent = opendir(FLAGS_input_dir.c_str());
    MACE_CHECK(dir_parent != nullptr, "Open input_dir ", FLAGS_input_dir,
               " failed: ", strerror(errno));
    int input_file_count = 0;
    std::string prefix = FormatName(input_names[0]);
    while ((entry = readdir(dir_parent))) {
      std::string file_name = std::string(entry->d_name);
      if (file_name.find(prefix) == 0) {
        ++input_file_count;
        std::string suffix = file_name.substr(prefix.size());

        for (size_t i = 0; i < input_count; ++i) {
          file_name =
              FLAGS_input_dir + "/" + FormatName(input_names[i]) + suffix;
          ReadInputDataFromFile(
              file_name, inputs_size[input_names[i]],
              input_data_types[i], inputs[input_names[i]].data<char>());
        }
        engine->Run(inputs, &outputs);

        if (!FLAGS_output_dir.empty()) {
          for (size_t i = 0; i < output_count; ++i) {
            std::string output_name =
                FLAGS_output_dir + "/" + FormatName(output_names[i]) + suffix;
            auto output_data_type = outputs[output_names[i]].data_type();
            auto file_data_type =
                (output_data_type == IDT_INT32) ? IDT_INT32 : IDT_FLOAT;
            auto output_size = WriteOutputDataToFile(
                output_name, file_data_type,
                outputs[output_names[i]].data<void>(),
                output_data_type, output_shapes[i]);
            LOG(INFO) << "Write output file " << output_name << " with size "
                      << output_size << " done.";
          }
        }
      }
    }

    closedir(dir_parent);
    MACE_CHECK(
        input_file_count != 0, "Found no input file name starts with \'",
        prefix, "\' in: ", FLAGS_input_dir,
        ", input file name should start with input tensor name.");
  } else {
    LOG(INFO) << "Warm up run";
    double warmup_millis;
    while (true) {
      int64_t t3 = NowMicros();
      MaceStatus warmup_status = engine->Run(inputs, &outputs);
      if (warmup_status != MaceStatus::MACE_SUCCESS) {
        LOG(ERROR) << "Warmup runtime error, retry ... errcode: "
                   << warmup_status.information();
        do {
#ifdef MODEL_GRAPH_FORMAT_CODE
          create_engine_status =
            CreateMaceEngineFromCode(model_name,
                                     reinterpret_cast<const unsigned char *>(
                                       model_weights_data->data()),
                                     model_weights_data->length(),
                                     input_names,
                                     output_names,
                                     config,
                                     &engine,
                                     model_data_unused,
                                     tutor,
                                     FLAGS_fake_warmup);
#else
          create_engine_status =
              CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                            model_graph_data->data()),
                                        model_graph_data->length(),
                                        reinterpret_cast<const unsigned char *>(
                                            model_weights_data->data()),
                                        model_weights_data->length(),
                                        input_names,
                                        output_names,
                                        config,
                                        &engine,
                                        model_data_unused,
                                        tutor,
                                        FLAGS_fake_warmup);
#endif
        } while (create_engine_status != MaceStatus::MACE_SUCCESS);
      } else {
        int64_t t4 = NowMicros();
        warmup_millis = (t4 - t3) / 1000.0;
        LOG(INFO) << "1st warm up run latency: " << warmup_millis << " ms";
        break;
      }
    }

    double model_run_millis = -1;
    benchmark::OpStat op_stat;
    if (FLAGS_round > 0) {
      LOG(INFO) << "Run model";
      int64_t total_run_duration = 0;
      for (int i = 0; i < FLAGS_round; ++i) {
        std::unique_ptr<port::Logger> info_log;
        std::unique_ptr<port::MallocLogger> malloc_logger;
        if (FLAGS_malloc_check_cycle >= 1
            && i % FLAGS_malloc_check_cycle == 0) {
          info_log = LOG_PTR(INFO);
          malloc_logger = port::Env::Default()->NewMallocLogger(
              info_log.get(), MakeString(i));
        }
        MaceStatus run_status;
        RunMetadata metadata;
        RunMetadata *metadata_ptr = nullptr;
        if (FLAGS_benchmark) {
          metadata_ptr = &metadata;
        }

        while (true) {
          int64_t t0 = NowMicros();
          run_status = engine->Run(inputs, &outputs, metadata_ptr);
          if (run_status != MaceStatus::MACE_SUCCESS) {
            LOG(ERROR) << "Mace run model runtime error, retry ... errcode: "
                       << run_status.information();
            do {
#ifdef MODEL_GRAPH_FORMAT_CODE
              create_engine_status =
                CreateMaceEngineFromCode(
                    model_name,
                    reinterpret_cast<const unsigned char *>(
                      model_weights_data->data()),
                    model_weights_data->length(),
                    input_names,
                    output_names,
                    config,
                    &engine,
                    model_data_unused,
                    tutor,
                    FLAGS_fake_warmup);
#else
              create_engine_status =
                  CreateMaceEngineFromProto(
                      reinterpret_cast<const unsigned char *>(
                          model_graph_data->data()),
                      model_graph_data->length(),
                      reinterpret_cast<const unsigned char *>(
                          model_weights_data->data()),
                      model_weights_data->length(),
                      input_names,
                      output_names,
                      config,
                      &engine,
                      model_data_unused,
                      tutor,
                      FLAGS_fake_warmup);
#endif
            } while (create_engine_status != MaceStatus::MACE_SUCCESS);
          } else {
            int64_t t1 = NowMicros();
            total_run_duration += (t1 - t0);
            if (FLAGS_benchmark) {
              op_stat.StatMetadata(metadata);
            }
            break;
          }
        }
      }
      model_run_millis = total_run_duration / 1000.0 / FLAGS_round;
      LOG(INFO) << "Average latency: " << model_run_millis << " ms";
    }

    for (size_t i = 0; i < output_count; ++i) {
      std::string output_name =
          FLAGS_output_file + "_" + FormatName(output_names[i]);
      auto output_data_type = outputs[output_names[i]].data_type();
      auto file_data_type =
          output_data_type == IDT_INT32 ? IDT_INT32 : IDT_FLOAT;

      auto output_size = WriteOutputDataToFile(
          output_name, file_data_type, outputs[output_names[i]].data<void>(),
          output_data_type, output_shapes[i]);
      LOG(INFO) << "Write output file " << output_name << " with size "
                << output_size << " done.";
    }

    // Metrics reporting tools depends on the format, keep in consistent
    printf("========================================================\n");
    printf("     capability(CPU)        init      warmup     run_avg\n");
    printf("========================================================\n");
    printf("time %15.3f %11.3f %11.3f %11.3f\n",
           cpu_capability, init_millis, warmup_millis, model_run_millis);
    if (FLAGS_benchmark) {
      op_stat.PrintStat();
    }
  }

  return true;
}

int Main(int argc, char **argv) {
  LOG(INFO) << "hello";
  std::string usage = "MACE run model tool, please specify proper arguments.\n"
                      "usage: " + std::string(argv[0])
      + " --help";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  if (input_names.empty() || output_names.empty()) {
    LOG(INFO) << gflags::ProgramUsage();
    return 0;
  }

  if (FLAGS_benchmark) {
    setenv("MACE_OPENCL_PROFILING", "1", 1);
    setenv("MACE_HEXAGON_PROFILING", "1", 1);
    setenv("MACE_QNN_PROFILE_LEVEL", "2", 2);
  }

  LOG(INFO) << "model name: " << FLAGS_model_name;
  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "input data_type: " << FLAGS_input_data_type;
  LOG(INFO) << "input data_format: " << FLAGS_input_data_format;
  //LOG(INFO) << "output node: " << FLAGS_output_node;
  //LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input data_type: " << FLAGS_output_data_type;
  LOG(INFO) << "output data_format: " << FLAGS_output_data_format;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "input dir: " << FLAGS_input_dir;
  LOG(INFO) << "output dir: " << FLAGS_output_dir;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "model_file: " << FLAGS_model_file;
  LOG(INFO) << "accelerator_cache_policy: " << FLAGS_accelerator_cache_policy;
  LOG(INFO) << "accelerator_binary_file: " << FLAGS_accelerator_binary_file;
  LOG(INFO) << "accelerator_storage_file: " << FLAGS_accelerator_storage_file;
  LOG(INFO) << "apu_boost_hint: " << FLAGS_apu_boost_hint;
  LOG(INFO) << "apu_preference_hint: " << FLAGS_apu_preference_hint;
  LOG(INFO) << "round: " << FLAGS_round;
  LOG(INFO) << "restart_round: " << FLAGS_restart_round;
  LOG(INFO) << "gpu_perf_hint: " << FLAGS_gpu_perf_hint;
  LOG(INFO) << "gpu_priority_hint: " << FLAGS_gpu_priority_hint;
  LOG(INFO) << "num_threads: " << FLAGS_num_threads;
  LOG(INFO) << "cpu_affinity_policy: " << FLAGS_cpu_affinity_policy;
  
  auto limit_opencl_kernel_time = getenv("MACE_LIMIT_OPENCL_KERNEL_TIME");
  if (limit_opencl_kernel_time) {
    LOG(INFO) << "limit_opencl_kernel_time: "
              << limit_opencl_kernel_time;
  }
  auto opencl_queue_window_size = getenv("MACE_OPENCL_QUEUE_WINDOW_SIZE");
  if (opencl_queue_window_size) {
    LOG(INFO) << "opencl_queue_window_size: "
              << getenv("MACE_OPENCL_QUEUE_WINDOW_SIZE");
  }

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
  if (input_names.size() != input_shape_vec.size()
      || output_names.size() != output_shape_vec.size()) {
    LOG(INFO) << "inputs' names do not match inputs' shapes "
                 "or outputs' names do not match outputs' shapes";
    return 0;
  }

  auto raw_input_data_types = Split(FLAGS_input_data_type, ',');
  std::vector<IDataType> input_data_types(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_data_types[i] = ParseDataType(raw_input_data_types[i]);
  }

  auto raw_output_data_types = Split(FLAGS_output_data_type, ',');
  std::vector<IDataType> output_data_types(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    output_data_types[i] = ParseDataType(raw_output_data_types[i]);
    //LOG(INFO) << "raw_output_data_types[" << i << "] is "
    //          << raw_output_data_types[i];
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
  float cpu_float32_performance = 0.0f;
  if (FLAGS_input_dir.empty()) {
    // get cpu capability
    Capability cpu_capability =
        GetCapability(static_cast<DeviceType>(RuntimeType::RT_CPU));
    cpu_float32_performance = cpu_capability.float32_performance.exec_time;
  }
  bool ret = false;
  
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    VLOG(0) << "restart round " << i;
    ret = RunModel(FLAGS_model_name, input_names, input_shape_vec,
                   input_data_types, input_data_formats, output_names,
                   output_shape_vec, output_data_types, output_data_formats,
                   cpu_float32_performance);
  }
  if (ret) {
    return 0;
  }
  return -1;
}

}  // namespace tools
}  // namespace mace

int main(int argc, char **argv) {
  mace::tools::Main(argc, argv);
}
