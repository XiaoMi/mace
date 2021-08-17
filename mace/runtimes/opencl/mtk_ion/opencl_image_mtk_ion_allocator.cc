// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/mtk_ion/opencl_image_mtk_ion_allocator.h"

#include <memory>

#include "mace/core/runtime_failure_mock.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_util.h"
#include "mace/runtimes/opencl/opencl_image_allocator.h"
#include "mace/utils/logging.h"

namespace mace {

OpenclImageMtkIonAllocator::OpenclImageMtkIonAllocator(
    OpenclExecutor *opencl_executor, std::shared_ptr<Rpcmem> rpcmem)
    : OpenclBaseMtkIonAllocator(opencl_executor, rpcmem) {
  clGetDeviceInfo(opencl_executor_->device().get(), CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(pitch_), &pitch_, nullptr);
}

MemoryType OpenclImageMtkIonAllocator::GetMemType() {
  return MemoryType::GPU_IMAGE;
}

MaceStatus OpenclImageMtkIonAllocator::New(const MemInfo &info, void **result) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_IMAGE);
  MACE_LATENCY_LOGGER(1, "Allocate OpenCL ION image: ",
                      info.dims[0], ", ", info.dims[1]);

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  void *host = nullptr;
  auto width = info.dims[0];
  auto height = info.dims[1];
  cl::ImageFormat img_format(CL_RGBA, OpenCLUtil::DataTypeToCLChannelType(info.data_type));
  cl_int error = CL_SUCCESS;
  cl_mem ion_mem;
  int nbytes = GetEnumTypeSize(info.data_type) * 4;
  size_t row_pitch = (width + pitch_ - 1) & ~(pitch_ - 1);

  CreateMtkIONPtr(cpu_ion_allocator_.get(), row_pitch * height * nbytes, &host, &ion_mem, &error);
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << width << ", " << height << "] failed because of "
                 << OpenCLErrorToString(error);
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  desc.image_array_size = 0;
  desc.image_row_pitch = row_pitch * nbytes;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = ion_mem;
  cl_mem image = clCreateImage(
    opencl_executor_->context().get(),
    0,  // flags inherited from buffer
    &img_format,
    &desc,
    NULL,
    &error);
  cl::Image2D *cl_image = new cl::Image2D(image);

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
    cl_to_host_map_[static_cast<void *>(cl_image)] = host;
    *result = cl_image;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenclImageMtkIonAllocator::Delete(void *image) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL image");
  if (image != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(image);
    delete cl_image;

    auto it = cl_to_host_map_.find(image);
    MACE_CHECK(it != cl_to_host_map_.end(), "OpenCL image not found!");
    rpcmem_->Delete(it->second);
    cl_to_host_map_.erase(image);
  }
}

}  // namespace mace
