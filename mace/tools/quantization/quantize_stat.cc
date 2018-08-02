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

/**
 * Usage:
 * quantize_stat --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_dir=input_data_dir \
 *          --output_file=mace.out  \
 *          --model_data_file=model_data.data
 */
#include <malloc.h>
#include <dirent.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"
#include "mace/utils/utils.h"

#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

namespace mace {
namespace tools {
namespace quantization {

namespace str_util {

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  if (str.empty()) {
    result.push_back("");
    return result;
  }
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

DEFINE_string(model_name,
              "",
              "model name in yaml");
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
DEFINE_string(input_dir,
              "",
              "input directory name");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
DEFINE_string(model_file,
              "",
              "model file name, used when load mace model in pb");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");

bool RunModel(const std::string &model_name,
              const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes) {
  MACE_RETURN_IF_ERROR(mace::SetOpenMPThreadPolicy(
      FLAGS_omp_num_threads, CPUAffinityPolicy::AFFINITY_NONE));

  std::vector<unsigned char> model_pb_data;
  if (FLAGS_model_file != "") {
    if (!mace::ReadBinaryFile(&model_pb_data, FLAGS_model_file)) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
    }
  }

  std::shared_ptr<mace::MaceEngine> engine;

  // Create Engine
#ifdef MODEL_GRAPH_FORMAT_CODE
  MACE_RETURN_IF_ERROR(
        CreateMaceEngineFromCode(model_name,
                                 FLAGS_model_data_file,
                                 input_names,
                                 output_names,
                                 DeviceType::CPU,
                                 &engine));
#else
  (void) (model_name);
  MACE_RETURN_IF_ERROR(
      CreateMaceEngineFromProto(model_pb_data,
                                FLAGS_model_data_file,
                                input_names,
                                output_names,
                                DeviceType::CPU,
                                &engine));
#endif

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  std::map<std::string, int64_t> inputs_size;
  for (size_t i = 0; i < input_count; ++i) {
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    inputs_size[input_names[i]] = input_size;
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
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

  DIR *dir_parent;
  struct dirent *entry;
  dir_parent = opendir(FLAGS_input_dir.c_str());
  if (dir_parent) {
    while ((entry = readdir(dir_parent))) {
      std::string file_name = std::string(entry->d_name);
      std::string prefix = FormatName(input_names[0]);
      if (file_name.find(prefix) == 0) {
        std::string suffix = file_name.substr(prefix.size());

        for (size_t i = 0; i < input_count; ++i) {
          file_name = FLAGS_input_dir + "/" + FormatName(input_names[i])
              + suffix;
          std::ifstream in_file(file_name, std::ios::in | std::ios::binary);
          VLOG(2) << "Read " << file_name;
          if (in_file.is_open()) {
            in_file.read(reinterpret_cast<char *>(
                             inputs[input_names[i]].data().get()),
                         inputs_size[input_names[i]] * sizeof(float));
            in_file.close();
          } else {
            LOG(INFO) << "Open input file failed";
            return -1;
          }
        }
        MACE_RETURN_IF_ERROR(engine->Run(inputs, &outputs));
      }
    }

    closedir(dir_parent);
  } else {
    LOG(ERROR) << "Directory " << FLAGS_input_dir << " does not exist.";
  }
  return true;
}

int Main(int argc, char **argv) {
  std::string usage = "quantize stat model\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "model name: " << FLAGS_model_name;
  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "output node: " << FLAGS_output_node;
  LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_dir: " << FLAGS_input_dir;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "model_file: " << FLAGS_model_file;
  LOG(INFO) << "omp_num_threads: " << FLAGS_omp_num_threads;

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

  return RunModel(FLAGS_model_name, input_names, input_shape_vec,
                  output_names, output_shape_vec);
}

}  // namespace quantization
}  // namespace tools
}  // namespace mace

int main(int argc, char **argv) {
  mace::tools::quantization::Main(argc, argv);
}
