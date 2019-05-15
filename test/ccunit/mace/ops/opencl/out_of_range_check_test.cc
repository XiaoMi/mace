// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#include "mace/core/op_context.h"
#include "mace/core/runtime/opencl/gpu_device.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/ops/opencl/helper.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace {

MaceStatus BufferToImageOpImpl(OpContext *context,
                               Tensor *buffer,
                               Tensor *image,
                               const std::vector<size_t> &image_shape) {
  std::unique_ptr<BufferBase> oorc_flag;
  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();

  std::string kernel_name = "in_out_buffer_to_image";
  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  MACE_OUT_OF_RANGE_CONFIG;
  MACE_NON_UNIFORM_WG_CONFIG;
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
  MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_to_image",
                                            obfuscated_kernel_name,
                                            built_options,
                                            &kernel));
  MACE_OUT_OF_RANGE_INIT(kernel);
  uint32_t idx = 0;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel.setArg(idx++,
                  *(static_cast<cl::Buffer *>(oorc_flag->buffer())));
  }
  MACE_SET_2D_GWS_ARGS(kernel, gws);
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

  cl_int error;
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
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  runtime->command_queue().finish();
  bool is_out_of_range = false;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    oorc_flag->Map(nullptr);
    is_out_of_range = *(oorc_flag->mutable_data<char>()) == 1 ? true : false;
    oorc_flag->UnMap();
  }
  return is_out_of_range ? MaceStatus::MACE_OUT_OF_RESOURCES
                         : MaceStatus::MACE_SUCCESS;
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

  GPUContext gpu_context;
  std::unique_ptr<Device> device = make_unique<GPUDevice>(
      gpu_context.opencl_tuner());

  Workspace ws;
  OpContext context(&ws, device.get());

  std::vector<index_t> buffer_shape = {batch, height, width, channels};
  Tensor *buffer =
      ws.CreateTensor("Buffer", device->allocator(),
                      DataTypeToEnum<float>::v());
  buffer->Resize(buffer_shape);

  std::vector<size_t> image_shape;
  Tensor *image = ws.CreateTensor("Image", device->allocator(),
                                  DataTypeToEnum<float>::v());
  OpenCLUtil::CalImage2DShape(buffer->shape(),
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  image->ResizeImage(buffer->shape(), image_shape);
  ASSERT_FALSE(BufferToImageOpImpl(&context, buffer, image, image_shape)
                   != MaceStatus::MACE_SUCCESS);

  std::vector<size_t> overflow_image_shape = image_shape;
  for (size_t i = 0; i < overflow_image_shape.size(); ++i) {
    overflow_image_shape[i] += 1;
  }
  ASSERT_TRUE(BufferToImageOpImpl(&context,
                                  buffer,
                                  image,
                                  overflow_image_shape)
                  != MaceStatus::MACE_SUCCESS);
}

}  // namespace ops
}  // namespace mace
