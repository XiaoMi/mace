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

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {
namespace {

bool BufferToImageOpImpl(Tensor *buffer,
                         Tensor *image,
                         const std::vector<size_t> &image_shape) {
  std::unique_ptr<BufferBase> kernel_error;
  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};

  auto runtime = OpenCLRuntime::Global();

  std::string kernel_name = "in_out_buffer_to_image";
  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  OUT_OF_RANGE_CONFIG(kernel_error);
  NON_UNIFORM_WG_CONFIG;
  if (buffer->dtype() == image->dtype()) {
    built_options.emplace("-DDATA_TYPE=" +
                          DtToCLDt(DataTypeToEnum<float>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(DataTypeToEnum<float>::value));
  } else {
    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpCompatibleCLDt(DataTypeToEnum<float>::value));
    built_options.emplace(
        "-DCMD_DATA_TYPE=" +
            DtToUpCompatibleCLCMDDt(DataTypeToEnum<float>::value));
  }

  cl::Kernel kernel;
  cl_int error = runtime->BuildKernel("buffer_to_image",
                                      obfuscated_kernel_name,
                                      built_options,
                                      &kernel);
  if (error != CL_SUCCESS) {
    return false;
  }

  uint32_t idx = 0;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel.setArg(idx++,
                  *(static_cast<cl::Buffer *>(kernel_error->buffer())));
  }
  SET_2D_GWS_ARGS(kernel);
  kernel.setArg(idx++, *(buffer->opencl_buffer()));
  MACE_CHECK(buffer->buffer_offset() % GetEnumTypeSize(buffer->dtype()) == 0,
             "buffer offset not aligned");
  kernel.setArg(idx++,
                    static_cast<uint32_t>(buffer->buffer_offset() /
                                          GetEnumTypeSize(buffer->dtype())));
  kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
  kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
  kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
  kernel.setArg(idx++, *(image->opencl_image()));

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(gws[0], gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }

    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  }
  if (error != CL_SUCCESS) {
    return false;
  }

  runtime->command_queue().finish();
  bool is_out_of_range = false;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_error->Map(nullptr);
    is_out_of_range = *(kernel_error->mutable_data<char>()) == 1 ? true : false;
    kernel_error->UnMap();
  }
  return is_out_of_range;
}

}  // namespace

class OutOfRangeCheckTest : public ::testing::Test {
 protected:
  virtual void SetUp() { setenv("OUT_OF_RANGE_CHECK", "1", 1); }
};

TEST(OutOfRangeCheckTest, RandomTest) {
  index_t batch = 3;
  index_t height = 5;
  index_t width = 7;
  index_t channels = 11;

  std::vector<index_t> buffer_shape = {batch, height, width, channels};
  Workspace ws;
  Tensor *buffer =
      ws.CreateTensor("Buffer", GetDeviceAllocator(DeviceType::GPU),
                      DataTypeToEnum<float>::v());
  buffer->Resize(buffer_shape);

  std::vector<size_t> image_shape;
  Tensor *image = ws.CreateTensor("Image", GetDeviceAllocator(DeviceType::GPU),
                                  DataTypeToEnum<float>::v());
  CalImage2DShape(buffer->shape(), IN_OUT_CHANNEL, &image_shape);
  image->ResizeImage(buffer->shape(), image_shape);
  ASSERT_FALSE(BufferToImageOpImpl(buffer, image, image_shape));

  std::vector<size_t> overflow_image_shape = image_shape;
  for (size_t i = 0; i < overflow_image_shape.size(); ++i) {
    overflow_image_shape[i] += 1;
  }
  ASSERT_TRUE(BufferToImageOpImpl(buffer, image, overflow_image_shape));
}

}  // namespace kernels
}  // namespace mace
