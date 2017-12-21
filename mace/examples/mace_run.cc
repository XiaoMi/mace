//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=MobilenetV1/Logits/conv2d/convolution  \
 *          --input_shape=1,3,224,224   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --device=NEON
 */
#include <fstream>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include "mace/utils/command_line_flags.h"
#include "mace/core/mace.h"
#include "mace/utils/logging.h"
#include "mace/utils/env_time.h"

using namespace std;
using namespace mace;

namespace mace {
extern NetDef MACE_MODEL_FUNCTION();
}
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

DeviceType ParseDeviceType(const string &device_str) {
  if(device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("NEON") == 0) {
    return DeviceType::NEON;
  } else if (device_str.compare("OPENCL") == 0) {
    return DeviceType::OPENCL;
  } else {
    return DeviceType::CPU;
  }
}

int main(int argc, char **argv) {
  string model_file;
  string input_node;
  string output_node;
  string input_shape;
  string input_file;
  string output_file;
  string device;
  int round = 1;

  std::vector<Flag> flag_list = {
      Flag("model", &model_file, "model file name"),
      Flag("input", &input_node, "input node"),
      Flag("output", &output_node, "output node"),
      Flag("input_shape", &input_shape, "input shape, separated by comma"),
      Flag("input_file", &input_file, "input file name"),
      Flag("output_file", &output_file, "output file name"),
      Flag("device", &device, "CPU/NEON"),
      Flag("round", &round, "round"),
  };

  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  VLOG(0) << "model: " << model_file << std::endl
          << "input: " << input_node << std::endl
          << "output: " << output_node << std::endl
          << "input_shape: " << input_shape << std::endl
          << "input_file: " << input_file << std::endl
          << "output_file: " << output_file << std::endl
          << "device: " << device << std::endl
          << "round: " << round << std::endl;

  vector<int64_t> shape;
  ParseShape(input_shape, &shape);

  // load model
  int64_t t0 = utils::NowMicros();
  NetDef net_def = mace::MACE_MODEL_FUNCTION();
  int64_t t1 = utils::NowMicros();
  LOG(INFO) << "CreateNetDef duration: " << t1 - t0 << " us";
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(device);
  VLOG(1) << "Device Type" << device_type;
  int64_t input_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);

  // load input
  ifstream in_file(input_file, ios::in | ios::binary);
  in_file.read(reinterpret_cast<char *>(input_data.get()),
               input_size * sizeof(float));
  in_file.close();

  // Init model
  VLOG(0) << "Run init";
  t0 = utils::NowMicros();
  mace::MaceEngine engine(&net_def, device_type);
  t1 = utils::NowMicros();
  init_micros += t1 - t0;
  LOG(INFO) << "Net init duration: " << t1 - t0 << " us";

  LOG(INFO) << "Total init duration: " << init_micros << " us";

  std::vector<int64_t> output_shape;
  VLOG(0) << "Warm up";
  t0 = utils::NowMicros();
  engine.Run(input_data.get(), shape, output_shape);
  t1 = utils::NowMicros();
  LOG(INFO) << "1st warm up run duration: " << t1 - t0 << " us";

  if (round > 0) {
    VLOG(0) << "Run model";
    t0 = utils::NowMicros();
    for (int i = 0; i < round; ++i) {
      engine.Run(input_data.get(), shape, output_shape);
    }
    t1 = utils::NowMicros();
    LOG(INFO) << "Avg duration: " << (t1 - t0) / round << " us";
  }

  const float *output = engine.Run(input_data.get(), shape, output_shape);
  if (output != nullptr) {
    ofstream out_file(output_file, ios::binary);
    int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    out_file.write((const char *) (output),
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    stringstream ss;
    ss << "Output shape: [";
    for (auto i : output_shape) {
      ss << i << ", ";
    }
    ss << "]";
    VLOG(0) << ss.str();
  }
}
