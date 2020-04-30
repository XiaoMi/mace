// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/opencl_image_allocator.h"

#include "mace/core/runtime_failure_mock.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_util.h"
#include "mace/utils/logging.h"

namespace mace {

MemoryType OpenclImageAllocator::GetMemType() {
  return MemoryType::GPU_IMAGE;
}

MaceStatus OpenclImageAllocator::New(const MemInfo &info,
                                     void **result) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_IMAGE);
  auto width = info.dims[0];
  auto height = info.dims[1];
  MACE_CHECK(width != 0 && height != 0,
      "Image shape's size must equal 2 and dim must not be 0.");
  MACE_LATENCY_LOGGER(1, "Allocate OpenCL image: ", width, ", ", height);

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl::ImageFormat img_format(
      CL_RGBA, OpenCLUtil::DataTypeToCLChannelType(info.data_type));
  cl_int error = CL_SUCCESS;
  cl::Image2D *cl_image = new cl::Image2D(
      opencl_executor_->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      img_format, width, height, 0, nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << width << ", " << height << "] failed because of "
                 << OpenCLErrorToString(error);
    // Many users have doubts at CL_INVALID_IMAGE_SIZE, add some tips.
    if (error == CL_INVALID_IMAGE_SIZE) {
      auto max_2d_size = opencl_executor_->GetMaxImage2DSize();
      LOG(WARNING) << "The allowable OpenCL image size is: "
                   << max_2d_size[0] << "x" << max_2d_size[1];
    }
    delete cl_image;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    *result = cl_image;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenclImageAllocator::Delete(void *image) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL image");
  if (image != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(image);
    delete cl_image;
  }
}

}  // namespace mace
