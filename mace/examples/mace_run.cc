//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=MobilenetV1/Logits/conv2d/convolution  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --device=NEON
 */
#include <malloc.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include "mace/utils/command_line_flags.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

#include "mace/core/public/mace.h"
#include "mace/core/public/version.h"

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
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("NEON") == 0) {
    return DeviceType::NEON;
  } else if (device_str.compare("OPENCL") == 0) {
    return DeviceType::OPENCL;
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

int main(int argc, char **argv) {
  string model_file;
  string input_node;
  string output_node;
  string input_shape;
  string output_shape;
  string input_file;
  string output_file;
  string device;
  int round = 1;
  int malloc_check_cycle = -1;

  std::vector<Flag> flag_list = {
      Flag("model", &model_file, "model file name"),
      Flag("input", &input_node, "input node"),
      Flag("output", &output_node, "output node"),
      Flag("input_shape", &input_shape, "input shape, separated by comma"),
      Flag("output_shape", &output_shape, "output shape, separated by comma"),
      Flag("input_file", &input_file, "input file name"),
      Flag("output_file", &output_file, "output file name"),
      Flag("device", &device, "CPU/NEON"),
      Flag("round", &round, "round"),
      Flag("malloc_check_cycle", &malloc_check_cycle,
           "malloc debug check cycle, -1 to disable"),
  };

  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  VLOG(0) << "mace version: " << MaceVersion() << std::endl
          << "mace git version: " << MaceGitVersion() << std::endl
          << "model: " << model_file << std::endl
          << "input: " << input_node << std::endl
          << "output: " << output_node << std::endl
          << "input_shape: " << input_shape << std::endl
          << "output_shape: " << output_shape << std::endl
          << "input_file: " << input_file << std::endl
          << "output_file: " << output_file << std::endl
          << "device: " << device << std::endl
          << "round: " << round << std::endl;

  vector<int64_t> input_shape_vec;
  vector<int64_t> output_shape_vec;
  ParseShape(input_shape, &input_shape_vec);
  ParseShape(output_shape, &output_shape_vec);

  // load model
  int64_t t0 = utils::NowMicros();
  NetDef net_def = mace::MACE_MODEL_FUNCTION();
  int64_t t1 = utils::NowMicros();
  LOG(INFO) << "CreateNetDef duration: " << t1 - t0 << " us";
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(device);
  VLOG(1) << "Device Type" << device_type;
  int64_t input_size = std::accumulate(input_shape_vec.begin(),
      input_shape_vec.end(), 1, std::multiplies<int64_t>());
  int64_t output_size = std::accumulate(output_shape_vec.begin(),
      output_shape_vec.end(), 1, std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> output_data(new float[output_size]);

  // load input
  ifstream in_file(input_file, ios::in | ios::binary);
  if (in_file.is_open()) {
    in_file.read(reinterpret_cast<char *>(input_data.get()),
                 input_size * sizeof(float));
    in_file.close();
  } else {
    LOG(ERROR) << "Open input file failed";
  }

  // Init model
  VLOG(0) << "Run init";
  t0 = utils::NowMicros();
  mace::MaceEngine engine(&net_def, device_type);
  t1 = utils::NowMicros();
  init_micros += t1 - t0;
  LOG(INFO) << "Net init duration: " << t1 - t0 << " us";

  LOG(INFO) << "Total init duration: " << init_micros << " us";

  VLOG(0) << "Warm up";
  t0 = utils::NowMicros();
  engine.Run(input_data.get(), input_shape_vec, output_data.get());
  t1 = utils::NowMicros();
  LOG(INFO) << "1st warm up run duration: " << t1 - t0 << " us";

  if (round > 0) {
    VLOG(0) << "Run model";
    t0 = utils::NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < round; ++i) {
      engine.Run(input_data.get(), input_shape_vec, output_data.get());
      if (malloc_check_cycle >= 1 && i % malloc_check_cycle == 0) {
        LOG(INFO) << "=== check malloc info change #" << i << " ===";
        prev = LogMallinfoChange(prev);
      }
    }
    t1 = utils::NowMicros();
    LOG(INFO) << "Avg duration: " << (t1 - t0) / round << " us";
  }

  if (output_data != nullptr) {
    ofstream out_file(output_file, ios::binary);
    out_file.write((const char *) (output_data.get()),
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file done.";
  } else {
    LOG(ERROR) << "output data is null";
  }
}
