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

#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/resize_bicubic.h"

namespace mace {
namespace ops {
namespace test {

class ResizeBicubicTest : public OpsTestBase {};

TEST_F(ResizeBicubicTest, CPUResizeBicubicWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 6, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, CPUResizeBicubicWOAlignCorners1) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(147);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 7, 7, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {4, 5})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 4, 5, 3},
      {0.,          1.,          2.,          4.272914,    5.2729135,
       6.272914, 8.255402,    9.255403,   10.255403,   12.744597,
       13.744597,   14.744598, 17.05098,    18.05098,    19.05098,
       35.765625,   36.765625,   37.765625, 40.038536,   41.038536,
       42.038536,   44.021027,   45.021027,   46.021027, 48.510223,
       49.51022,    50.51022,    52.816605,   53.816605,   54.816605, 73.5,
       74.5,        75.5,        77.77291,    78.77291,    79.77292,
       81.7554,     82.7554,     83.7554,     86.2446,     87.24459,
       88.2446, 90.55097,    91.55098,    92.55098,   111.97266,   112.97266,
       113.97266, 116.24557,   117.24557,   118.24557,   120.22806,
       121.228065,  122.22806, 124.71725,   125.71725,   126.717255,
       129.02362,   130.02364,   131.02364});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, CPUResizeBicubicWOAlignCorners2) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(243);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 9, 9, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {7, 6})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 7, 6, 3},
      {0.,         1.,         2.,         4.5,        5.5,        6.5,
      9.,        10.,        11.,        13.5,       14.5,       15.5,
      18.,        19.,        20.,        22.78125,   23.78125,   24.78125,
      35.90507,   36.905067,  37.905067,  40.40507,   41.40507,   42.40507,
      44.90507,   45.905067,  46.90507,   49.40507,   50.40507,   51.40507,
      53.90507,   54.90507,   55.90507,   58.68632,   59.68632,   60.686314,
      68.953384,  69.953384,  70.953384,  73.45339,   74.453384,  75.453384,
      77.953384,  78.95339,   79.953384,  82.453384,  83.453384,  84.453384,
      86.953384,  87.95338,   88.953384,  91.734634,  92.73464,   93.73464,
      102.97063,  103.970634, 104.97063,  107.470634, 108.470634, 109.47064,
      111.970634, 112.970634, 113.97063,  116.470634, 117.47065,  118.47065,
      120.970634, 121.970634, 122.970634, 125.751884, 126.751884, 127.75187,
      140.02936,  141.02936,  142.02936,  144.52939,  145.52936,  146.52936,
      149.02936,  150.02937,  151.02937,  153.52936,  154.52936,  155.52936,
      158.02937,  159.02937,  160.02937,  162.81061,  163.81062,  164.81061,
      174.04663,  175.04662,  176.04662,  178.54663,  179.54663,  180.54662,
      183.0466,   184.0466,   185.04662,  187.54663,  188.54663,  189.54663,
      192.04662,  193.0466,   194.04662,  196.82788,  197.82788,  198.82787,
      210.0477,   211.04767,  212.04768,  214.54767,  215.54768,  216.54767,
      219.0477,   220.04767,  221.04768,  223.54767,  224.54767,  225.54767,
      228.04768,  229.04767,  230.04767,  232.82892,  233.82892,  234.8289});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, CPUResizeBicubicWOAlignCornersFloat) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(48);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 4, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {2, 3})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 2, 3, 3},
      {0., 1., 2., 4.110297, 5.110297, 6.110297,
       8.223037, 9.223036, 10.223037, 24., 25., 26.,
       28.110298, 29.1103, 30.110298, 32.223038, 33.223038, 34.223038});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, ResizeBicubicWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("align_corners", 1)
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 9, 10, 11});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, CPUResizeBicubicWAlignCorners1) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(147);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 7, 7, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("align_corners", 1)
      .AddIntsArg("size", {4, 5})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 4, 5, 3},
      {0.,    1.,    2.,    4.5,   5.5,   6.5,
       9.,   10.,   11.,   13.5,  14.5,  15.5,
       18.,   19.,   20.,   42.,   43.,   44.,
       46.5,  47.5,  48.5,  51.,   52.,   53.,
       55.5,  56.5,  57.5,  60.,   61.,   62.,
       84.,   85.,   86.,   88.5,  89.5,  90.5,
       93.,   94.,   95.,   97.5,  98.5,  99.5,
       102.,  103.,  104.,  126.,  127.,  128.,
       130.5, 131.5, 132.5, 135.,  136.,  137.,
       139.5, 140.5, 141.5, 144.,  145.,  146.});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBicubicTest, CPUResizeBicubicWAlignCorners2) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(243);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 9, 9, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("align_corners", 1)
      .AddIntsArg("size", {7, 6})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  auto expected = CreateTensor<float>({1, 7, 6, 3},
      {0.,          1.,          2.,          4.727086,    5.727086,    6.727086,
      9.7445965,  10.7445965,  11.744598,   14.255403,   15.255402,   16.255402,
      19.272913,   20.272913,   21.272913,   24.,         25.,         26.,
      36.992676,   37.992676,   38.992676,   41.71976,    42.71976,    43.71976,
      46.73727,    47.737274,   48.737274,   51.248077,   52.248077,   53.248077,
      56.265587,   57.265587,   58.265587,   60.99268,    61.992676,   62.992676,
      71.007324,   72.007324,   73.007324,   75.734406,   76.73441,    77.734406,
      80.75193,    81.75194,    82.75193,    85.262726,   86.262726,   87.262726,
      90.28025,    91.28024,    92.28024,    95.00733,    96.007324,   97.007324,
      108.,        109.,        110.,        112.72709,   113.72708,   114.72708,
      117.7446,    118.7446,    119.7446,    122.2554,    123.2554,    124.2554,
      127.2729,    128.27292,   129.27292,   132.,        133.,        134.,
      144.99268,   145.99268,   146.99268,   149.71977,   150.71974,   151.71976,
      154.73726,   155.73727,   156.73727,   159.24808,   160.24808,   161.2481,
      164.26556,   165.2656,    166.2656,    168.99269,   169.99268,   170.99269,
      179.00732,   180.00732,   181.00732,   183.73439,   184.73444,   185.7344,
      188.75192,   189.75192,   190.75192,   193.26273,   194.26274,   195.26274,
      198.28024,   199.28024,   200.28026,   203.00734,   204.00732,   205.00734,
      216.,        217.,        218.,        220.72707,   221.72707,   222.72708,
      225.74458,   226.7446,    227.74458,   230.25539,   231.2554,    232.2554,
      235.2729,    236.2729,    237.27289,   240.,        241.,        242.});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void TestRandomResizeBicubic() {
  testing::internal::LogToStderr();
  static unsigned int seed = time(NULL);
  for (int round = 0; round < 10; ++round) {
    int batch = 1 + rand_r(&seed) % 5;
    int channels = 1 + rand_r(&seed) % 100;
    int height = 1 + rand_r(&seed) % 100;
    int width = 1 + rand_r(&seed) % 100;
    int in_height = 1 + rand_r(&seed) % 100;
    int in_width = 1 + rand_r(&seed) % 100;
    int align_corners = rand_r(&seed) % 1;

    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, in_height, in_width, channels});
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);

    OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());
    // Run on CPU
    net.RunOp(DeviceType::CPU);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    if (D == DeviceType::GPU) {
      BufferToImage<D, float>(&net, "Input", "InputImage",
                              kernels::BufferType::IN_OUT_CHANNEL);

      OpDefBuilder("ResizeBicubic", "ResizeBicubicTest")
          .Input("InputImage")
          .Output("OutputImage")
          .AddIntArg("align_corners", align_corners)
          .AddIntsArg("size", {height, width})
          .Finalize(net.NewOperatorDef());
      // Run
      net.RunOp(D);

      ImageToBuffer<D, float>(&net, "OutputImage", "DeviceOutput",
                              kernels::BufferType::IN_OUT_CHANNEL);
    }
    // Check
    ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 1e-5,
                            1e-4);
  }
}
}  // namespace

TEST_F(ResizeBicubicTest, OPENCLRandomResizeBicubic) {
  TestRandomResizeBicubic<DeviceType::GPU>();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
