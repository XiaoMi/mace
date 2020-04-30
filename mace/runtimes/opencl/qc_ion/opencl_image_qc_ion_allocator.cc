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

#include "mace/runtimes/opencl/qc_ion/opencl_image_qc_ion_allocator.h"

#include <memory>

#include "mace/core/runtime_failure_mock.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_util.h"
#include "mace/runtimes/opencl/opencl_image_allocator.h"
#include "mace/utils/logging.h"

namespace mace {

OpenclImageQcIonAllocator::OpenclImageQcIonAllocator(
    OpenclExecutor *opencl_executor, std::shared_ptr<Rpcmem> rpcmem)
    : OpenclBaseQcIonAllocator(opencl_executor, rpcmem) {}

MemoryType OpenclImageQcIonAllocator::GetMemType() {
  return MemoryType::GPU_IMAGE;
}

MaceStatus OpenclImageQcIonAllocator::New(const MemInfo &info, void **result) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_IMAGE);
  MACE_LATENCY_LOGGER(1, "Allocate OpenCL ION image: ",
                      info.dims[0], ", ", info.dims[1]);

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl::ImageFormat img_format(
      CL_RGBA, OpenCLUtil::DataTypeToCLChannelType(info.data_type));
  cl_int error = CL_SUCCESS;

  cl_mem_ion_host_ptr ion_host;
  size_t pitch;
  CreateQualcommImageIONHostPtr(info.dims, img_format, &pitch, &ion_host);

  cl::Image2D *cl_image = new cl::Image2D(
      opencl_executor_->context(),
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
      img_format,
      info.dims[0],
      info.dims[1],
      pitch,
      &ion_host,
      &error);

  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << info.dims[0] << ", " << info.dims[1]
                 << "] failed because of "
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
    cl_to_host_map_[static_cast<void *>(cl_image)] = ion_host.ion_hostptr;
    *result = cl_image;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenclImageQcIonAllocator::Delete(void *image) {
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

void OpenclImageQcIonAllocator::CreateQualcommImageIONHostPtr(
    const std::vector<index_t> &shape,
    const cl::ImageFormat &format,
    size_t *pitch,
    cl_mem_ion_host_ptr *ion_host) {
  cl_int error = clGetDeviceImageInfoQCOM(
      opencl_executor_->device().get(), shape[0], shape[1], &format,
      CL_IMAGE_ROW_PITCH, sizeof(*pitch), pitch, nullptr);
  MACE_CHECK(error == CL_SUCCESS, "clGetDeviceImageInfoQCOM failed, error: ",
             OpenCLErrorToString(error));

  CreateQualcommBufferIONHostPtr(cpu_ion_allocator_.get(),
                                 *pitch * shape[1], ion_host);
}

}  // namespace mace
