//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/ops_test_util.h"
#include "mace/kernels/conv_pool_2d_util.h"

using namespace mace;

class AtrousConv2dOpTest : public OpsTestBase {};

static void UpSampleFilter(const std::vector<index_t> &filter_shape,
                           const std::vector<float> &filter_data,
                           const int dilation_rate,
                           std::vector<index_t> &upsampled_filter_shape,
                           std::vector<float> &upsampled_filter_data) {
  upsampled_filter_shape[0] = filter_shape[0];
  upsampled_filter_shape[1] = filter_shape[1];
  upsampled_filter_shape[2] = filter_shape[2] + (filter_shape[2] - 1) * (dilation_rate - 1);
  upsampled_filter_shape[3] = filter_shape[3] + (filter_shape[3] - 1) * (dilation_rate - 1);
  const index_t upsampled_filter_size = std::accumulate(upsampled_filter_shape.begin(),
                                                        upsampled_filter_shape.end(),
                                                        1, std::multiplies<index_t>());
  upsampled_filter_data.resize(upsampled_filter_size, 0);
  index_t filter_idx = 0;
  index_t upsampled_filter_idx = 0;
  for (index_t n = 0; n < filter_shape[0]; ++n) {
    for (index_t c = 0; c < filter_shape[1]; ++c) {
      for (index_t h = 0; h < filter_shape[2]; ++h) {
        for (index_t w = 0; w < filter_shape[3]; ++w) {
          upsampled_filter_data[upsampled_filter_idx] = filter_data[filter_idx];
          filter_idx += 1;
          upsampled_filter_idx += dilation_rate;
        }
        upsampled_filter_idx += 1 - dilation_rate + (dilation_rate-1) * upsampled_filter_shape[3];
      }
      upsampled_filter_idx -= (dilation_rate-1) * upsampled_filter_shape[3];
    }
  }
}

template <DeviceType D>
static void RunConv2D(const std::vector<index_t> &input_shape,
                const std::vector<float> &input_data,
                const std::vector<index_t> &filter_shape,
                const std::vector<float> &filter_data,
                const std::vector<index_t> &bias_shape,
                const std::vector<float> &bias_data,
                const int dilation_h,
                const int dilation_w,
                Padding padding,
                Tensor *result) {
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {dilation_h, dilation_w})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", input_shape, input_data);
  net.AddInputFromArray<D, float>(
      "Filter", filter_shape, filter_data);
  net.AddInputFromArray<D, float>("Bias", bias_shape, bias_data);

  // Run
  net.RunOp(D);

  // Check
  result->Copy(*net.GetOutput("Output"));
}

template <DeviceType D>
static void GenerateAndRunConv2D(const index_t batch,
                                 const index_t input_channels,
                                 const index_t height,
                                 const index_t width,
                                 const index_t output_channels,
                                 const index_t kernel_h,
                                 const index_t kernel_w,
                                 Padding  padding,
                                 const int dilation_rate)  {
  srand(time(NULL));
  // Add input data
  std::vector<index_t> input_shape = {batch, input_channels, height, width};
  std::vector<float> input_data;
  GenerateRandomRealTypeData<float>(input_shape, input_data);
  std::vector<index_t> filter_shape = {output_channels, input_channels, kernel_h, kernel_w};
  std::vector<float> filter_data;
  GenerateRandomRealTypeData<float>(filter_shape, filter_data);
  std::vector<index_t> bias_shape = {output_channels};
  std::vector<float> bias_data;
  GenerateRandomRealTypeData<float>(bias_shape, bias_data);

  std::vector<index_t> upsampled_filter_shape(4, 0);
  std::vector<float> upsampled_filter_data;
  UpSampleFilter(filter_shape, filter_data, dilation_rate,
                 upsampled_filter_shape, upsampled_filter_data);
  Tensor expected_result;
  // Run on cpu
  RunConv2D<DeviceType::CPU>(input_shape, input_data,
                             upsampled_filter_shape, upsampled_filter_data,
                             bias_shape, bias_data,
                             1, 1,
                             padding, &expected_result);

  Tensor device_result(GetDeviceAllocator(D), DataTypeToEnum<float>::v());
  // run on device
  RunConv2D<D>(input_shape, input_data,
               filter_shape, filter_data,
               bias_shape, bias_data,
               dilation_rate, dilation_rate,
               padding, &device_result);
  ExpectTensorNear<float>(expected_result, device_result, 0.001);
}
template <DeviceType D>
static void TestSimple(const int kernel_h,
                        const int kernel_w,
                        Padding  padding,
                        const int dilation_rate) {
  GenerateAndRunConv2D<D>(1, 3, 5, 5, 1, kernel_h, kernel_w, padding, dilation_rate);
}

TEST_F(AtrousConv2dOpTest, CPUSimple) {
  for (int i = 2 ; i < 4; ++i) {
    TestSimple<DeviceType::CPU>(3, 3, VALID, i);
    TestSimple<DeviceType::CPU>(3, 3, SAME, i);
  }
}

TEST_F(AtrousConv2dOpTest, OPENCLSimple) {
  for (int i = 2 ; i < 3; ++i) {
    TestSimple<DeviceType::OPENCL>(3, 3, VALID, i);
  }
}

template <DeviceType D>
static void TestAligned(const int kernel_h,
                        const int kernel_w,
                        Padding  padding,
                        const int dilation_rate) {
  GenerateAndRunConv2D<D>(3, 64, 32, 32, 128, kernel_h, kernel_w, padding, dilation_rate);
}

template <DeviceType D>
static void TestUnAligned(const int kernel_h,
                        const int kernel_w,
                        Padding  padding,
                        const int dilation_rate) {
  srand(time(NULL));
  // generate random input
  index_t batch = 3 + rand() % 10;
  index_t input_channels = 3 + rand() % 10;
  index_t height = 107;
  index_t width = 113;
  index_t output_channels = 3 + rand() % 10;

  GenerateAndRunConv2D<D>(batch, input_channels, height, width, output_channels,
                       kernel_h, kernel_w, padding, dilation_rate);
}

TEST_F(AtrousConv2dOpTest, UpSample) {
  const int batch = 2;
  const int channel = 2;
  const int height = 3;
  const int width = 3;
  const int rate = 2;
  std::vector<index_t> filter_shape = {batch, channel, height, width};
  std::vector<float> filter_data(batch*channel*height*width, 1);
  std::vector<index_t> upsampled_filter_shape(4, 0);
  std::vector<float> upsampled_filter_data;
  UpSampleFilter(filter_shape, filter_data, rate,
                 upsampled_filter_shape, upsampled_filter_data);
  int size = std::accumulate(upsampled_filter_shape.begin(), upsampled_filter_shape.end(),
                             1, std::multiplies<index_t>());
  const int expected_size = batch * channel *
      (height + (height-1) * (rate - 1)) *
      (width + (width-1) * (rate-1));
  EXPECT_EQ(expected_size, upsampled_filter_data.size());
}


TEST_F(AtrousConv2dOpTest, CPUAligned) {
  for (int i = 2 ; i < 4; ++i) {
    TestAligned<DeviceType::CPU>(3, 3, VALID, i);
    TestAligned<DeviceType::CPU>(3, 3, SAME, i);
  }
}

TEST_F(AtrousConv2dOpTest, OPENCLAligned) {
  for (int i = 2 ; i < 4; ++i) {
    TestAligned<DeviceType::OPENCL>(3, 3, VALID, i);
    TestAligned<DeviceType::OPENCL>(3, 3, SAME, i);
  }
}

TEST_F(AtrousConv2dOpTest, CPUUnAligned) {
  for (int i = 2 ; i < 4; ++i) {
    TestUnAligned<DeviceType::CPU>(3, 3, VALID, i);
    TestUnAligned<DeviceType::CPU>(3, 3, SAME, i);
  }
}

