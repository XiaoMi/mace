//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --model_data_file=model_data.data \
 *          --device=OPENCL
 */
#include <malloc.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

using namespace std;
using namespace mace;

namespace mace {
namespace MACE_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_MODEL_TAG
}  // namespace mace


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

void ParseShape(const string &str, vector<int64_t> *shape) {
  string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == string::npos) {
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

DeviceType ParseDeviceType(const string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("NEON") == 0) {
    return DeviceType::NEON;
  } else if (device_str.compare("OPENCL") == 0) {
    return DeviceType::OPENCL;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else {
    return DeviceType::CPU;
  }
}

struct mallinfo LogMallinfoChange(struct mallinfo prev) {
  struct mallinfo curr = mallinfo();
  if (prev.arena != curr.arena) {
    LOG(INFO) << "Non-mmapped space allocated (bytes): " << curr.arena
              << ", diff: " << ((int64_t)curr.arena - (int64_t)prev.arena);
  }
  if (prev.ordblks != curr.ordblks) {
    LOG(INFO) << "Number of free chunks: " << curr.ordblks
              << ", diff: " << ((int64_t)curr.ordblks - (int64_t)prev.ordblks);
  }
  if (prev.smblks != curr.smblks) {
    LOG(INFO) << "Number of free fastbin blocks: " << curr.smblks
              << ", diff: " << ((int64_t)curr.smblks - (int64_t)prev.smblks);
  }
  if (prev.hblks != curr.hblks) {
    LOG(INFO) << "Number of mmapped regions: " << curr.hblks
              << ", diff: " << ((int64_t)curr.hblks - (int64_t)prev.hblks);
  }
  if (prev.hblkhd != curr.hblkhd) {
    LOG(INFO) << "Space allocated in mmapped regions (bytes): " << curr.hblkhd
              << ", diff: " << ((int64_t)curr.hblkhd - (int64_t)prev.hblkhd);
  }
  if (prev.usmblks != curr.usmblks) {
    LOG(INFO) << "Maximum total allocated space (bytes): " << curr.usmblks
              << ", diff: " << ((int64_t)curr.usmblks - (int64_t)prev.usmblks);
  }
  if (prev.fsmblks != curr.fsmblks) {
    LOG(INFO) << "Space in freed fastbin blocks (bytes): " << curr.fsmblks
              << ", diff: " << ((int64_t)curr.fsmblks - (int64_t)prev.fsmblks);
  }
  if (prev.uordblks != curr.uordblks) {
    LOG(INFO) << "Total allocated space (bytes): " << curr.uordblks
              << ", diff: "
              << ((int64_t)curr.uordblks - (int64_t)prev.uordblks);
  }
  if (prev.fordblks != curr.fordblks) {
    LOG(INFO) << "Total free space (bytes): " << curr.fordblks << ", diff: "
              << ((int64_t)curr.fordblks - (int64_t)prev.fordblks);
  }
  if (prev.keepcost != curr.keepcost) {
    LOG(INFO) << "Top-most, releasable space (bytes): " << curr.keepcost
              << ", diff: "
              << ((int64_t)curr.keepcost - (int64_t)prev.keepcost);
  }
  return curr;
}

DEFINE_string(input_node, "input_node0,input_node1", "input nodes, separated by comma");
DEFINE_string(input_shape, "1,224,224,3:1,1,1,10", "input shapes, separated by colon and comma");
DEFINE_string(output_node, "output_node0,output_node1", "output nodes, separated by comma");
DEFINE_string(output_shape, "1,224,224,2:1,1,1,10", "output shapes, separated by colon and comma");
DEFINE_string(input_file, "", "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file, "", "output file name | output file prefix for multiple outputs");
DEFINE_string(model_data_file, "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_string(device, "OPENCL", "CPU/NEON/OPENCL/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");

bool SingleInputAndOutput(const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &output_shape) {
  // load model
  int64_t t0 = NowMicros();
  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);
  int64_t t1 = NowMicros();
  LOG(INFO) << "CreateNetDef latency: " << t1 - t0 << " us";
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(FLAGS_device);
  LOG(INFO) << "Runing with device type: " << device_type;

  // Init model
  LOG(INFO) << "Run init";
  t0 = NowMicros();
  mace::MaceEngine engine(&net_def, device_type);
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }
  t1 = NowMicros();
  init_micros += t1 - t0;
  LOG(INFO) << "Net init latency: " << t1 - t0 << " us";
  LOG(INFO) << "Total init latency: " << init_micros << " us";

  // Allocate input and output
  int64_t input_size =
      std::accumulate(input_shape.begin(), input_shape.end(), 1,
                      std::multiplies<int64_t>());
  int64_t output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), 1,
                      std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> output_data(new float[output_size]);

  // load input
  ifstream in_file(FLAGS_input_file + "_" + FormatName(FLAGS_input_node), ios::in | ios::binary);
  if (in_file.is_open()) {
    in_file.read(reinterpret_cast<char *>(input_data.get()),
                 input_size * sizeof(float));
    in_file.close();
  } else {
    LOG(INFO) << "Open input file failed";
    return -1;
  }

  LOG(INFO) << "Warm up run";
  t0 = NowMicros();
  engine.Run(input_data.get(), input_shape, output_data.get());
  t1 = NowMicros();
  LOG(INFO) << "1st warm up run latency: " << t1 - t0 << " us";

  if (FLAGS_round > 0) {
    LOG(INFO) << "Run model";
    t0 = NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < FLAGS_round; ++i) {
      engine.Run(input_data.get(), input_shape, output_data.get());
      if (FLAGS_malloc_check_cycle >= 1 && i % FLAGS_malloc_check_cycle == 0) {
        LOG(INFO) << "=== check malloc info change #" << i << " ===";
        prev = LogMallinfoChange(prev);
      }
    }
    t1 = NowMicros();
    LOG(INFO) << "Averate latency: " << (t1 - t0) / FLAGS_round << " us";
  }

  if (output_data != nullptr) {
    std::string output_name = FLAGS_output_file + "_" + FormatName(FLAGS_output_node);
    ofstream out_file(output_name, ios::binary);
    out_file.write((const char *) (output_data.get()),
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file "
              << output_name
              << " with size " << output_size
              << " done.";
  } else {
    LOG(INFO) << "Output data is null";
  }

  return true;
}

bool MultipleInputOrOutput(const std::vector<std::string> &input_names,
                           const std::vector<std::vector<int64_t>> &input_shapes,
                           const std::vector<std::string> &output_names,
                           const std::vector<std::vector<int64_t>> &output_shapes) {
  // load model
  int64_t t0 = NowMicros();
  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);
  int64_t t1 = NowMicros();
  LOG(INFO) << "CreateNetDef latency: " << t1 - t0 << " us";
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(FLAGS_device);
  LOG(INFO) << "Runing with device type: " << device_type;

  // Init model
  LOG(INFO) << "Run init";
  t0 = NowMicros();
  mace::MaceEngine engine(&net_def, device_type, input_names, output_names);
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }
  t1 = NowMicros();
  init_micros += t1 - t0;
  LOG(INFO) << "Net init latency: " << t1 - t0 << " us";
  LOG(INFO) << "Total init latency: " << init_micros << " us";

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();
  std::vector<mace::MaceInputInfo> input_infos(input_count);
  std::map<std::string, float*> outputs;
  std::vector<std::unique_ptr<float[]>> input_datas(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    input_datas[i].reset(new float[input_size]);
    // load input
    ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]), ios::in | ios::binary);
    if (in_file.is_open()) {
      in_file.read(reinterpret_cast<char *>(input_datas[i].get()),
                   input_size * sizeof(float));
      in_file.close();
    } else {
      LOG(INFO) << "Open input file failed";
      return -1;
    }
    input_infos[i].name = input_names[i];
    input_infos[i].shape = input_shapes[i];
    input_infos[i].data = input_datas[i].get();
  }
  std::vector<std::unique_ptr<float[]>> output_datas(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    output_datas[i].reset(new float[output_size]);
    outputs[output_names[i]] = output_datas[i].get();
  }

  LOG(INFO) << "Warm up run";
  t0 = NowMicros();
  engine.Run(input_infos, outputs);
  t1 = NowMicros();
  LOG(INFO) << "1st warm up run latency: " << t1 - t0 << " us";

  if (FLAGS_round > 0) {
    LOG(INFO) << "Run model";
    t0 = NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < FLAGS_round; ++i) {
      engine.Run(input_infos, outputs);
      if (FLAGS_malloc_check_cycle >= 1 && i % FLAGS_malloc_check_cycle == 0) {
        LOG(INFO) << "=== check malloc info change #" << i << " ===";
        prev = LogMallinfoChange(prev);
      }
    }
    t1 = NowMicros();
    LOG(INFO) << "Averate latency: " << (t1 - t0) / FLAGS_round << " us";
  }

  for (size_t i = 0; i < output_count; ++i) {
    std::string output_name = FLAGS_output_file + "_" + FormatName(output_names[i]);
    ofstream out_file(output_name, ios::binary);
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    out_file.write((const char *) outputs[output_names[i]],
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file "
              << output_name
              << " with size " << output_size
              << " done.";
  }

  return true;
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "mace git version: " << MaceGitVersion();
  LOG(INFO) << "model checksum: " << mace::MACE_MODEL_TAG::ModelChecksum();
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "output node: " << FLAGS_output_node;
  LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "device: " << FLAGS_device;
  LOG(INFO) << "round: " << FLAGS_round;

  std::vector<std::string> input_names = str_util::Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = str_util::Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes = str_util::Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = str_util::Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<vector<int64_t>> input_shape_vec(input_count);
  std::vector<vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }

  bool ret;
  if (input_count == 1 && output_count == 1) {
    ret = SingleInputAndOutput(input_shape_vec[0], output_shape_vec[0]);
  } else {
    ret = MultipleInputOrOutput(input_names, input_shape_vec, output_names, output_shape_vec);
  }
  if(ret) {
    return 0;
  } else {
    return -1;
  }
}
