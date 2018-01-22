//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {

class WinogradTransformOpTest : public OpsTestBase {};

//TEST_F(WinogradTransformOpTest, WinogradInputTransform) {
//  srand(time(NULL));
//
//  // generate random input
//  index_t batch = 7;
//  index_t height = 61;
//  index_t width = 71;
//  index_t channels = 31;
//
//  index_t p = batch * ((height - 1) / 2) * ((width - 1) / 2);
//
//  const std::string A_file = "/data/local/tmp/test/A";
//  const std::string C_file = "/data/local/tmp/test/C";
//  const std::vector<index_t> A_shape = {batch, height, width, channels};
//  const int A_size = std::accumulate(A_shape.begin(), A_shape.end(), 1, std::multiplies<int>());
//  const std::vector<index_t> C_shape = {16, channels, p, 1};
//  const int C_size = std::accumulate(C_shape.begin(), C_shape.end(), 1, std::multiplies<int>());
//
//  std::vector<float> A_data(A_size, 0.0);
//  std::ifstream in_file(A_file, std::ios::in | std::ios::binary);
//  if (in_file.is_open()) {
//    in_file.read(reinterpret_cast<char *>(A_data.data()),
//                 A_size * sizeof(float));
//    in_file.close();
//  } else {
//    VLOG(0) << "open A file failed";
//  }
//  auto C_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
//                                                DataTypeToEnum<float>::v()));
//  C_tensor->Resize(C_shape);
//  std::vector<float> C_data(C_size, 0.0);
//  std::ifstream C_in_file(C_file, std::ios::in | std::ios::binary);
//  if (C_in_file.is_open()) {
//    C_in_file.read(reinterpret_cast<char *>(C_data.data()),
//                   C_size * sizeof(float));
//    C_in_file.close();
//    Tensor::MappingGuard C_mapper(C_tensor.get());
//    float *batch_ptr = C_tensor->mutable_data<float>();
//    MACE_CHECK(static_cast<size_t>(C_tensor->size()) ==
//        C_data.size());
//    memcpy(batch_ptr, C_data.data(), C_data.size() * sizeof(float));
//  } else {
//    VLOG(0) << "open C file failed";
//  }
//  // Construct graph
//  OpsTestNet net;
//  // Add input data
//  net.AddInputFromArray<DeviceType::OPENCL, float>(
//      "A", A_shape, A_data);
//
//  // Run on opencl
//  BufferToImage<DeviceType::OPENCL, float>(net, "A", "AImage",
//                                           kernels::BufferType::IN_OUT_CHANNEL);
//
//  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
//      .Input("AImage")
//      .Output("OutputImage")
//      .Finalize(net.NewOperatorDef());
//
//  // Run on opencl
//  net.RunOp(DeviceType::OPENCL);
//  net.Sync();
//
//  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
//                                           kernels::BufferType::IN_OUT_HEIGHT);
//  ExpectTensorNear<float>(*(C_tensor.get()), *net.GetOutput("OPENCLOutput"), 1e-4);
//}
//
//TEST_F(WinogradTransformOpTest, FilterTransform) {
//  srand(time(NULL));
//
//  // generate random input
//  index_t out_chan = 31;
//  index_t in_chan = 31;
//  index_t height = 3;
//  index_t width = 3;
//
//  index_t p = (in_chan + 3) / 4;
//
//  const std::string A_file = "/data/local/tmp/test/filter_in";
//  const std::string C_file = "/data/local/tmp/test/filter_out";
//  const std::vector<index_t> A_shape = {out_chan, in_chan, height, width};
//  const int A_size = std::accumulate(A_shape.begin(), A_shape.end(), 1, std::multiplies<int>());
//  const std::vector<index_t> C_shape = {16, out_chan, in_chan, 1};
//  const int C_size = std::accumulate(C_shape.begin(), C_shape.end(), 1, std::multiplies<int>());
//
//  std::vector<float> A_data(A_size, 0.0);
//  std::ifstream in_file(A_file, std::ios::in | std::ios::binary);
//  if (in_file.is_open()) {
//    in_file.read(reinterpret_cast<char *>(A_data.data()),
//                 A_size * sizeof(float));
//    in_file.close();
//  } else {
//    VLOG(0) << "open A file failed";
//  }
//  auto C_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
//                                                DataTypeToEnum<float>::v()));
//  C_tensor->Resize(C_shape);
//  std::vector<float> C_data(C_size, 0.0);
//  std::ifstream C_in_file(C_file, std::ios::in | std::ios::binary);
//  if (C_in_file.is_open()) {
//    C_in_file.read(reinterpret_cast<char *>(C_data.data()),
//                   C_size * sizeof(float));
//    C_in_file.close();
//    Tensor::MappingGuard C_mapper(C_tensor.get());
//    float *batch_ptr = C_tensor->mutable_data<float>();
//    MACE_CHECK(static_cast<size_t>(C_tensor->size()) ==
//        C_data.size());
//    memcpy(batch_ptr, C_data.data(), C_data.size() * sizeof(float));
//  } else {
//    VLOG(0) << "open C file failed";
//  }
//  // Construct graph
//  OpsTestNet net;
//  // Add input data
//  net.AddInputFromArray<DeviceType::OPENCL, float>(
//      "A", A_shape, A_data);
//
//  // Run on opencl
//
//  OpDefBuilder("BufferToImage", "WinogradFilterTransformTest")
//      .Input("A")
//      .AddIntArg("buffer_type", kernels::WINOGRAD_FILTER)
//      .Output("OutputImage")
//      .Finalize(net.NewOperatorDef());
//
//  // Run on opencl
//  net.RunOp(DeviceType::OPENCL);
//
//  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
//                                           kernels::BufferType::WINOGRAD_FILTER);
//  ExpectTensorNear<float>(*(C_tensor.get()), *net.GetOutput("OPENCLOutput"), 1e-4);
//}
//
//
//TEST_F(WinogradTransformOpTest, WinogradInverseTransform) {
//  srand(time(NULL));
//
//  // generate random input
//  index_t n = 7;
//  index_t out_height = 59;
//  index_t out_width = 69;
//  index_t out_chan = 31;
//
//  index_t p = n * ((out_height + 1) / 2) * ((out_width + 1) / 2);
//
//  const std::string A_file = "/data/local/tmp/test/gemm";
//  const std::string C_file = "/data/local/tmp/test/res";
//  const std::vector<index_t> A_shape = {16, out_chan, p, 1};
//  const int A_size = std::accumulate(A_shape.begin(), A_shape.end(), 1, std::multiplies<int>());
//  const std::vector<index_t> C_shape = {n, out_height, out_width, out_chan};
//  const int C_size = std::accumulate(C_shape.begin(), C_shape.end(), 1, std::multiplies<int>());
//
//  std::vector<float> A_data(A_size, 0.0);
//  std::ifstream in_file(A_file, std::ios::in | std::ios::binary);
//  if (in_file.is_open()) {
//    in_file.read(reinterpret_cast<char *>(A_data.data()),
//                 A_size * sizeof(float));
//    in_file.close();
//  } else {
//    VLOG(0) << "open A file failed";
//  }
//  auto C_tensor = unique_ptr<Tensor>(new Tensor(GetDeviceAllocator(DeviceType::OPENCL),
//                                                DataTypeToEnum<float>::v()));
//  C_tensor->Resize(C_shape);
//  std::vector<float> C_data(C_size, 0.0);
//  std::ifstream C_in_file(C_file, std::ios::in | std::ios::binary);
//  if (C_in_file.is_open()) {
//    C_in_file.read(reinterpret_cast<char *>(C_data.data()),
//                   C_size * sizeof(float));
//    C_in_file.close();
//    Tensor::MappingGuard C_mapper(C_tensor.get());
//    float *batch_ptr = C_tensor->mutable_data<float>();
//    MACE_CHECK(static_cast<size_t>(C_tensor->size()) ==
//        C_data.size());
//    memcpy(batch_ptr, C_data.data(), C_data.size() * sizeof(float));
//  } else {
//    VLOG(0) << "open C file failed";
//  }
//  // Construct graph
//  OpsTestNet net;
//  // Add input data
//  net.AddInputFromArray<DeviceType::OPENCL, float>(
//      "A", A_shape, A_data);
//
//  // Run on opencl
//  BufferToImage<DeviceType::OPENCL, float>(net, "A", "AImage",
//                                           kernels::BufferType::IN_OUT_HEIGHT);
//
//  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
//      .Input("AImage")
//      .AddIntArg("batch", n)
//      .AddIntArg("height", out_height)
//      .AddIntArg("width", out_width)
//      .Output("OutputImage")
//      .Finalize(net.NewOperatorDef());
//
//  // Run on opencl
//  net.RunOp(DeviceType::OPENCL);
//  net.Sync();
//
//  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
//                                           kernels::BufferType::IN_OUT_CHANNEL);
//  ExpectTensorNear<float>(*(C_tensor.get()), *net.GetOutput("OPENCLOutput"), 1e-4);
//}

void TransposeFilter(const std::vector<float> &input,
                     const std::vector<index_t> &input_shape,
                     std::vector<float> &output) {
  output.resize(input.size());

  const float *input_ptr = input.data();
  for (index_t h = 0; h < input_shape[0]; ++h) {
    for (index_t w = 0; w < input_shape[1]; ++w) {
      for (index_t ic = 0; ic < input_shape[2]; ++ic) {
        for (index_t oc = 0; oc < input_shape[3]; ++oc) {
          int offset = ((oc * input_shape[2] + ic) * input_shape[0] + h) * input_shape[1] + w;
          output[offset] = *input_ptr;
          ++input_ptr;
        }
      }
    }
  }
}

template<DeviceType D, typename T>
void WinogradConvolution(const index_t batch,
                         const index_t height,
                         const index_t width,
                         const index_t in_channels,
                         const index_t out_channels,
                         const Padding padding) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  // Add input data
  std::vector<float> filter_data;
  std::vector<index_t> filter_shape = {3, 3, in_channels, out_channels};
  GenerateRandomRealTypeData<float>(filter_shape, filter_data);
  net.AddRandomInput<D, float>("Input", {batch, height, width, in_channels});
  net.AddInputFromArray<D, float>("Filter", filter_shape, filter_data);

  BufferToImage<D, T>(net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(net, "Filter", "FilterImage",
                      kernels::BufferType::FILTER);
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("InputImage")
      .Input("FilterImage")
      .Output("OutputImage")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, T>(net, "OutputImage", "ConvOutput",
                      kernels::BufferType::IN_OUT_CHANNEL);
  Tensor expected;
  expected.Copy(*net.GetOutput("ConvOutput"));
  auto output_shape = expected.shape();

  // Winograd convolution
  // transform filter
  std::vector<float> wino_filter_data;
  TransposeFilter(filter_data, filter_shape, wino_filter_data);
  net.AddInputFromArray<D, float>("WinoFilterData", {out_channels, in_channels, 3, 3}, wino_filter_data);
  BufferToImage<D, T>(net, "WinoFilterData", "WinoFilter", kernels::BufferType::WINOGRAD_FILTER);

  // transform input
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("WinoInput")
      .AddIntArg("padding", padding)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);

  // GEMM
  OpDefBuilder("GEMM", "GEMMTest")
      .Input("WinoFilter")
      .Input("WinoInput")
      .Output("WinoGemm")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  // Run on opencl
  net.RunOp(D);

  // Inverse transform
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("WinoGemm")
      .AddIntArg("batch", batch)
      .AddIntArg("height", output_shape[1])
      .AddIntArg("width", output_shape[2])
      .Output("WinoOutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);
  net.Sync();

  ImageToBuffer<D, float>(net, "WinoOutputImage", "WinoOutput",
                                           kernels::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("WinoOutput"), 1e-1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("WinoOutput"), 1e-4);
  }
}


TEST_F(WinogradTransformOpTest, Convolution) {
  WinogradConvolution<DeviceType::OPENCL, float>(1, 64, 64, 32, 32, Padding::VALID);
}

}
