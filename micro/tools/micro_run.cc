// Copyright 2020 The MACE Authors. All Rights Reserved.
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
 * micro_run --input=input_node  \
 *           --output=output_node  \
 *           --input_shape=1,224,224,3   \
 *           --output_shape=1,224,224,2   \
 *           --input_file=input_data \
 *           --output_file=micro.out
 */

#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include "gflags/gflags.h"
#include "micro/base/logging.h"
#include "micro/include/public/micro.h"
#include "micro/include/utils/macros.h"
#include "micro/port/api.h"

#ifndef MICRO_MODEL_NAME
#error Please specify model name in the command
#endif

namespace micro {
namespace MICRO_MODEL_NAME {
MaceStatus GetMicroEngineSingleton(MaceMicroEngine **engine);
}

namespace tools {
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

void ParseShape(const std::string &str, std::vector<int32_t> *shape) {
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

DEFINE_string(model_name, "", "model name in yaml");
DEFINE_string(input_node, "", "input nodes, separated by comma");
DEFINE_string(input_shape, "",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node, "", "output nodes, separated by comma");
DEFINE_string(output_shape, "",
              "output shapes, separated by colon and comma");
DEFINE_string(input_data_format, "NHWC",
              "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format, "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file, "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file, "",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(input_dir, "", "input directory name");
DEFINE_string(output_dir, "output", "output directory name");

DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_bool(benchmark, false, "enable benchmark op");

void GetOutputAndStoreToFile(MaceMicroEngine *micro_engine,
                             const std::vector<std::string> &output_names,
                             const std::string &prefix,
                             const std::string &suffix) {
  for (size_t i = 0; i < output_names.size(); ++i) {
    void *output_buffer = NULL;
    const int32_t *output_dims = NULL;
    uint32_t dim_size = 0;
    MaceStatus status =
        micro_engine->GetOutputData(i, &output_buffer, &output_dims, &dim_size);
    MACE_UNUSED(status);
    MACE_ASSERT1(status == MACE_SUCCESS && output_buffer != NULL,
                 "GetOutputData failed");
    std::string output_name = prefix + FormatName(output_names[i]) + suffix;
    std::ofstream out_file(output_name, std::ios::binary);
    MACE_ASSERT2(out_file.is_open(), "Open output file failed: ",
                 strerror(errno));
    int64_t output_size = std::accumulate(output_dims, output_dims + dim_size,
                                          sizeof(float),
                                          std::multiplies<int64_t>());
    out_file.write(reinterpret_cast<char *>(output_buffer),
                   output_size);
    MACE_ASSERT1(!out_file.bad(), "write file failed!");
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file " << output_name.c_str()
              << " with size " << output_size << " done.";
  }
}

bool RunModel(const std::vector<std::string> &input_names,
              const std::vector<std::vector<int32_t>> &input_shapes,
              const std::vector<DataFormat> &input_data_formats,
              const std::vector<std::string> &output_names,
              const std::vector<DataFormat> &output_data_formats) {
  // for future
  MACE_UNUSED(input_data_formats);
  MACE_UNUSED(output_data_formats);

  int64_t t0 = port::api::NowMicros();
  MaceMicroEngine *micro_engine = NULL;
  MaceStatus status = MICRO_MODEL_NAME::GetMicroEngineSingleton(&micro_engine);
  MACE_UNUSED(status);
  MACE_ASSERT(status == MACE_SUCCESS && micro_engine != NULL);
  int64_t t1 = port::api::NowMicros();
  double init_millis = (t1 - t0) / 1000.0;
  LOG(INFO) << "Total init latency: "
            << static_cast<float>(init_millis) << " ms";

  std::vector<std::shared_ptr<char>> inputs;
  std::vector<int32_t> input_sizes;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_sizes.push_back(std::accumulate(input_shapes[i].begin(),
                                          input_shapes[i].end(), sizeof(float),
                                          std::multiplies<int32_t>()));
    inputs.push_back(std::shared_ptr<char>(new char[input_sizes[i]],
                                           std::default_delete<char[]>()));
  }

  if (!FLAGS_input_dir.empty()) {
    DIR *dir_parent;
    struct dirent *entry;
    dir_parent = opendir(FLAGS_input_dir.c_str());
    if (dir_parent == NULL) {
      LOG(FATAL) << "Open input_dir " << FLAGS_input_dir.c_str()
                 << " failed: " << strerror(errno);
    }
    while ((entry = readdir(dir_parent))) {
      std::string file_name = std::string(entry->d_name);
      std::string prefix = FormatName(input_names[0]);
      if (file_name.find(prefix) == 0) {
        std::string suffix = file_name.substr(prefix.size());

        for (size_t i = 0; i < input_names.size(); ++i) {
          file_name = FLAGS_input_dir + "/" + FormatName(input_names[i])
              + suffix;
          std::ifstream in_file(file_name, std::ios::in | std::ios::binary);
          LOG(INFO) << "Read " << file_name.c_str();
          MACE_ASSERT2(in_file.is_open(), "Open input file failed: ",
                       strerror(errno));
          in_file.read(inputs[i].get(), input_sizes[i]);
          in_file.close();
          micro_engine->RegisterInputData(i, inputs[i].get(),
                                          input_shapes[i].data());
        }
        status = micro_engine->Run();
        MACE_ASSERT(status == MACE_SUCCESS);

        if (!FLAGS_output_dir.empty()) {
          GetOutputAndStoreToFile(micro_engine, output_names,
                                  FLAGS_output_dir + "/", suffix);
        }
      }
    }

    closedir(dir_parent);
  } else {
    for (size_t i = 0; i < input_names.size(); ++i) {
      // load input
      std::ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]),
                            std::ios::in | std::ios::binary);
      if (in_file.is_open()) {
        in_file.read(inputs[i].get(), input_sizes[i]);
        in_file.close();
      } else {
        LOG(INFO) << "Open input file failed";
        return -1;
      }
      micro_engine->RegisterInputData(i, inputs[i].get(),
                                      input_shapes[i].data());
    }

    LOG(INFO) << "Warm up run";
    int64_t t3 = port::api::NowMicros();
    status = micro_engine->Run();
    MACE_ASSERT1(status == MACE_SUCCESS, "run micro engine failed");
    int64_t t4 = port::api::NowMicros();
    double warmup_millis = (t4 - t3) / 1000.0;
    LOG(INFO) << "1st warm up run latency: "
              << static_cast<float>(warmup_millis) << " ms";

    double model_run_millis = -1;
    if (FLAGS_round > 0) {
      LOG(INFO) << "Run model";
      int64_t total_run_duration = 0;
      for (int i = 0; i < FLAGS_round; ++i) {
        int64_t t0 = port::api::NowMicros();
        // TODO(luxuhui): add metadata to benchmark op
        status = micro_engine->Run();
        MACE_ASSERT(status == MACE_SUCCESS);
        int64_t t1 = port::api::NowMicros();
        total_run_duration += (t1 - t0);
      }
      model_run_millis = total_run_duration / 1000.0 / FLAGS_round;
      LOG(INFO) << "Average latency: "
                << static_cast<float>(model_run_millis) << " ms";
    }
    GetOutputAndStoreToFile(micro_engine, output_names,
                            FLAGS_output_file + "_", "");

    // Metrics reporting tools depends on the format, keep in consistent
    printf("=============================================\n");
    printf("----        init       warmup     run_avg    \n");
    printf("=============================================\n");
    printf("time %11.3f %11.3f %11.3f\n",
           init_millis, warmup_millis, model_run_millis);
  }

  return true;
}

int Main(int argc, char **argv) {
  std::string usage = "MACE micro run model tool, please specify proper"
                      " arguments.\nusage: " + std::string(argv[0]) + " --help";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  if (input_names.empty() || output_names.empty()) {
    LOG(INFO) << gflags::ProgramUsage();
    return 0;
  }

  LOG(INFO) << "model name: " << FLAGS_model_name.c_str();
  LOG(INFO) << "input node: " << FLAGS_input_node.c_str();
  LOG(INFO) << "input shape: " << FLAGS_input_shape.c_str();
  LOG(INFO) << "output node: " << FLAGS_output_node.c_str();
  LOG(INFO) << "output shape: " << FLAGS_output_shape.c_str();
  LOG(INFO) << "input_file: " << FLAGS_input_file.c_str();
  LOG(INFO) << "output_file: " << FLAGS_output_file.c_str();
  LOG(INFO) << "input dir: " << FLAGS_input_dir.c_str();
  LOG(INFO) << "output dir: " << FLAGS_output_dir.c_str();
  LOG(INFO) << "round: " << FLAGS_round;
  LOG(INFO) << "restart_round: " << FLAGS_restart_round;

  std::vector<std::string> input_shapes = Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int32_t>> input_shape_vec(input_count);
  std::vector<std::vector<int32_t>> output_shape_vec(output_count);
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
  bool ret = false;
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    LOG(INFO) << "restart round " << i;

    ret = RunModel(input_names, input_shape_vec, input_data_formats,
                   output_names, output_data_formats);
  }
  if (ret) {
    return 0;
  }
  return -1;
}

}  // namespace tools
}  // namespace micro

int main(int argc, char **argv) {
  micro::tools::Main(argc, argv);
}
