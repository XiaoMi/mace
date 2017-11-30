//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

/**
 * Usage:
 * mace_dsp_run --model=mobi_mace.pb \
 *          --input_shape=1,3,224,224   \
 *          --input_file=input_data \
 *          --output_file=mace.out
 */
#include <sys/time.h>
#include <fstream>
#include "mace/dsp/hexagon_control_wrapper.h"
#include "mace/core/net.h"
#include "mace/utils/command_line_flags.h"

using namespace std;
using namespace mace;

void ParseShape(const string &str, vector<index_t> *shape) {
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

int main(int argc, char **argv) {
  string model_file;
  string input_shape;
  string input_file;
  string output_file;
  int round = 1;

  std::vector<Flag> flag_list = {
      Flag("model", &model_file, "model file name"),
      Flag("input_shape", &input_shape, "input shape, separated by comma"),
      Flag("input_file", &input_file, "input file name"),
      Flag("output_file", &output_file, "output file name"),
      Flag("round", &round, "round"),
  };

  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  VLOG(0) << "model: " << model_file << std::endl
          << "input_shape: " << input_shape << std::endl
          << "input_file: " << input_file << std::endl
          << "output_file: " << output_file << std::endl
          << "round: " << round << std::endl;

  vector<index_t> shape;
  ParseShape(input_shape, &shape);

  // load input
  Tensor input_tensor;
  input_tensor.Resize(shape);
  float *input_data = input_tensor.mutable_data<float>();
  ifstream in_file(input_file, ios::in | ios::binary);
  in_file.read(reinterpret_cast<char *>(input_data),
               input_tensor.size() * sizeof(float));
  in_file.close();

  // execute
  HexagonControlWrapper wrapper;
  VLOG(0) << "version: " << wrapper.GetVersion();
  wrapper.Init();
  wrapper.SetDebugLevel(0);
  wrapper.Config();
  VLOG(0) << wrapper.SetupGraph(model_file);
  wrapper.PrintGraph();

  Tensor output_tensor;
  timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < round; ++i) {
    VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  }
  gettimeofday(&tv2, NULL);
  cout << "avg duration: "
       << ((tv2.tv_sec - tv1.tv_sec) * 1000 +
           (tv2.tv_usec - tv1.tv_usec) / 1000) /
           round
       << endl;

  wrapper.GetPerfInfo();
  wrapper.PrintLog();
  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();

  // save output
  ofstream out_file(output_file, ios::binary);
  out_file.write((const char *) (output_tensor.data<float>()),
                 output_tensor.size() * sizeof(float));
  out_file.flush();
  out_file.close();
}