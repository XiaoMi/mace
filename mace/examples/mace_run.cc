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
#include <sys/time.h>
#include <fstream>
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

  vector<index_t> shape;
  ParseShape(input_shape, &shape);

  // load model
  ifstream file_stream(model_file, ios::in | ios::binary);
  NetDef net_def;
  net_def.ParseFromIstream(&file_stream);
  file_stream.close();

  DeviceType device_type;
  DeviceType_Parse(device, &device_type);
  VLOG(0) << device_type;
  Workspace ws;
  ws.LoadModelTensor(net_def, device_type);
  Tensor *input_tensor =
      ws.CreateTensor(input_node + ":0", GetDeviceAllocator(device_type), DT_FLOAT);
  input_tensor->Resize(shape);
  {
    Tensor::MappingGuard input_guard(input_tensor);
    float *input_data = input_tensor->mutable_data<float>();

    // load input
    ifstream in_file(input_file, ios::in | ios::binary);
    in_file.read(reinterpret_cast<char *>(input_data),
                 input_tensor->size() * sizeof(float));
    in_file.close();
  }

  // Init model
  auto net = CreateNet(net_def, &ws, device_type, OpMode::INIT);
  net->Run();

  // run model
  net = CreateNet(net_def, &ws, device_type);

  VLOG(0) << "warm up";
  // warm up
  for (int i = 0; i < 1; ++i) {
    net->Run();
  }

  VLOG(0) << "run";
  timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < round; ++i) {
    net->Run();
  }
  gettimeofday(&tv2, NULL);
  cout << "avg duration: "
       << ((tv2.tv_sec - tv1.tv_sec) * 1000 +
           (tv2.tv_usec - tv1.tv_usec) / 1000) /
              round
       << endl;

  // save output
  const Tensor *output = ws.GetTensor(output_node + ":0");

  Tensor::MappingGuard output_guard(output);
  ofstream out_file(output_file, ios::binary);
  out_file.write((const char *)(output->data<float>()),
                 output->size() * sizeof(float));
  out_file.flush();
  out_file.close();
  VLOG(0) << "Output shape: ["
          << output->dim(0) << ", "
          << output->dim(1) << ", "
          << output->dim(2) << ", "
          << output->dim(3) << "]";
}