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

#include "mace/core/runtime/opencl/opencl_allocator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace {

static cl_channel_type DataTypeToCLChannelType(const DataType t) {
  switch (t) {
    case DT_HALF:
      return CL_HALF_FLOAT;
    case DT_FLOAT:
      return CL_FLOAT;
    case DT_INT32:
      return CL_SIGNED_INT32;
    case DT_UINT8:
      return CL_UNSIGNED_INT32;
    default:
      LOG(FATAL) << "Image doesn't support the data type: " << t;
      return 0;
  }
}

}  // namespace

OpenCLAllocator::OpenCLAllocator(
    OpenCLRuntime *opencl_runtime): opencl_runtime_(opencl_runtime) {}

OpenCLAllocator::~OpenCLAllocator() {}

MaceStatus OpenCLAllocator::New(size_t nbytes, void **result) {
  if (nbytes == 0) {
    return MaceStatus::MACE_SUCCESS;
  }
  VLOG(3) << "Allocate OpenCL buffer: " << nbytes;

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl_int error = CL_SUCCESS;
  cl::Buffer *buffer = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    cl_mem_ion_host_ptr ion_host;
    CreateQualcommBufferIONHostPtr(nbytes, &ion_host);

    buffer = new cl::Buffer(
        opencl_runtime_->context(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        nbytes, &ion_host, &error);

    cl_to_host_map_[static_cast<void *>(buffer)] = ion_host.ion_hostptr;
  } else {
#endif  // MACE_ENABLE_RPCMEM
    buffer = new cl::Buffer(opencl_runtime_->context(),
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            nbytes, nullptr, &error);
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL Buffer with "
                 << nbytes << " bytes failed because of "
                 << OpenCLErrorToString(error);
    delete buffer;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    *result = buffer;
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus OpenCLAllocator::NewImage(const std::vector<size_t> &image_shape,
                                     const DataType dt,
                                     void **result) {
  MACE_CHECK(image_shape.size() == 2, "Image shape's size must equal 2");
  MACE_LATENCY_LOGGER(1, "Allocate OpenCL image: ",
                      image_shape[0], ", ", image_shape[1]);

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl::ImageFormat img_format(CL_RGBA, DataTypeToCLChannelType(dt));
  cl_int error = CL_SUCCESS;
  cl::Image2D *cl_image = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    cl_mem_ion_host_ptr ion_host;
    size_t pitch;
    CreateQualcommImageIONHostPtr(image_shape, img_format, &pitch, &ion_host);

    cl_image = new cl::Image2D(
        opencl_runtime_->context(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        img_format, image_shape[0], image_shape[1], pitch, &ion_host, &error);

    cl_to_host_map_[static_cast<void *>(cl_image)] = ion_host.ion_hostptr;
  } else {
#endif  // MACE_ENABLE_RPCMEM
    cl_image =
        new cl::Image2D(opencl_runtime_->context(),
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, img_format,
                        image_shape[0], image_shape[1], 0, nullptr, &error);
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << image_shape[0] << ", " << image_shape[1]
                 << "] failed because of "
                 << OpenCLErrorToString(error);
    // Many users have doubts at CL_INVALID_IMAGE_SIZE, add some tips.
    if (error == CL_INVALID_IMAGE_SIZE) {
      auto max_2d_size = opencl_runtime_->GetMaxImage2DSize();
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

void OpenCLAllocator::Delete(void *buffer) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL buffer");
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
#ifdef MACE_ENABLE_RPCMEM
    if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
      auto it = cl_to_host_map_.find(buffer);
      MACE_CHECK(it != cl_to_host_map_.end(), "OpenCL buffer not found!");
      rpcmem_.Delete(it->second);
      cl_to_host_map_.erase(buffer);
    }
#endif  // MACE_ENABLE_RPCMEM
  }
}

void OpenCLAllocator::DeleteImage(void *buffer) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL image");
  if (buffer != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(buffer);
    delete cl_image;
#ifdef MACE_ENABLE_RPCMEM
    if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
      auto it = cl_to_host_map_.find(buffer);
      MACE_CHECK(it != cl_to_host_map_.end(), "OpenCL image not found!");
      rpcmem_.Delete(it->second);
      cl_to_host_map_.erase(buffer);
    }
#endif  // MACE_ENABLE_RPCMEM
  }
}

void *OpenCLAllocator::Map(void *buffer,
                           size_t offset,
                           size_t nbytes,
                           bool finish_cmd_queue) {
  MACE_LATENCY_LOGGER(1, "Map OpenCL buffer");
  void *mapped_ptr = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    auto it = cl_to_host_map_.find(buffer);
    MACE_CHECK(it != cl_to_host_map_.end(), "Try to map unallocated Buffer!");
    mapped_ptr = it->second;

    if (finish_cmd_queue) {
      opencl_runtime_->command_queue().finish();
    }

    if (opencl_runtime_->qcom_host_cache_policy() ==
        CL_MEM_HOST_WRITEBACK_QCOM) {
      MACE_CHECK(rpcmem_.SyncCacheStart(mapped_ptr) == 0);
    }
  } else {
#endif  // MACE_ENABLE_RPCMEM
    MACE_UNUSED(finish_cmd_queue);
    auto cl_buffer = static_cast<cl::Buffer *>(buffer);
    auto queue = opencl_runtime_->command_queue();
    // TODO(heliangliang) Non-blocking call
    cl_int error;
    mapped_ptr =
        queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                               offset, nbytes, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
      LOG(ERROR) << "Map buffer failed, error: " << OpenCLErrorToString(error);
    }
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
  return mapped_ptr;
}

// TODO(liuqi) there is something wrong with half type.
void *OpenCLAllocator::MapImage(void *buffer,
                                const std::vector<size_t> &image_shape,
                                std::vector<size_t> *mapped_image_pitch,
                                bool finish_cmd_queue) {
  MACE_LATENCY_LOGGER(1, "Map OpenCL Image");
  MACE_CHECK(image_shape.size() == 2) << "Just support map 2d image";
  void *mapped_ptr = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    // TODO(libin): Set mapped_image_pitch if needed
    auto it = cl_to_host_map_.find(buffer);
    MACE_CHECK(it != cl_to_host_map_.end(), "Try to map unallocated Image!");
    mapped_ptr = it->second;

    if (finish_cmd_queue) {
      opencl_runtime_->command_queue().finish();
    }

    if (opencl_runtime_->qcom_host_cache_policy() ==
        CL_MEM_HOST_WRITEBACK_QCOM) {
      MACE_CHECK(rpcmem_.SyncCacheStart(mapped_ptr) == 0);
    }
  } else {
#endif  // MACE_ENABLE_RPCMEM
    MACE_UNUSED(finish_cmd_queue);
    auto cl_image = static_cast<cl::Image2D *>(buffer);
    std::array<size_t, 3> origin = {{0, 0, 0}};
    std::array<size_t, 3> region = {{image_shape[0], image_shape[1], 1}};

    mapped_image_pitch->resize(2);
    cl_int error;
    mapped_ptr = opencl_runtime_->command_queue().enqueueMapImage(
        *cl_image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
        mapped_image_pitch->data(), mapped_image_pitch->data() + 1, nullptr,
        nullptr, &error);
    if (error != CL_SUCCESS) {
      LOG(ERROR) << "Map Image failed, error: " << OpenCLErrorToString(error);
    }
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer, void *mapped_ptr) {
  MACE_LATENCY_LOGGER(1, "Unmap OpenCL buffer/Image");
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    if (opencl_runtime_->qcom_host_cache_policy() ==
        CL_MEM_HOST_WRITEBACK_QCOM) {
      MACE_CHECK(rpcmem_.SyncCacheEnd(mapped_ptr) == 0);
    }
  } else {
#endif  // MACE_ENABLE_RPCMEM
    auto cl_buffer = static_cast<cl::Buffer *>(buffer);
    auto queue = opencl_runtime_->command_queue();
    cl_int error = queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr,
                                              nullptr, nullptr);
    if (error != CL_SUCCESS) {
      LOG(ERROR) << "Unmap buffer failed, error: "
                 << OpenCLErrorToString(error);
    }
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
}

bool OpenCLAllocator::OnHost() const { return false; }

#ifdef MACE_ENABLE_RPCMEM
Rpcmem *OpenCLAllocator::rpcmem() {
  return &rpcmem_;
}

void OpenCLAllocator::CreateQualcommBufferIONHostPtr(
    const size_t nbytes,
    cl_mem_ion_host_ptr *ion_host) {
  void *host = rpcmem_.New(nbytes + opencl_runtime_->qcom_ext_mem_padding());
  MACE_CHECK_NOTNULL(host);
  auto host_addr = reinterpret_cast<std::uintptr_t>(host);
  auto page_size = opencl_runtime_->qcom_page_size();
  MACE_CHECK(host_addr % page_size == 0, "ION memory address: ", host_addr,
             " must be aligned to page size: ", page_size);
  int fd = rpcmem_.ToFd(host);
  MACE_CHECK(fd >= 0, "Invalid rpcmem file descriptor: ", fd);

  ion_host->ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_host->ext_host_ptr.host_cache_policy =
      opencl_runtime_->qcom_host_cache_policy();
  ion_host->ion_filedesc = fd;
  ion_host->ion_hostptr = host;
}

void OpenCLAllocator::CreateQualcommImageIONHostPtr(
    const std::vector<size_t> &shape,
    const cl::ImageFormat &format,
    size_t *pitch,
    cl_mem_ion_host_ptr *ion_host) {
  cl_int error = clGetDeviceImageInfoQCOM(
      opencl_runtime_->device().get(), shape[0], shape[1], &format,
      CL_IMAGE_ROW_PITCH, sizeof(*pitch), pitch, nullptr);
  MACE_CHECK(error == CL_SUCCESS, "clGetDeviceImageInfoQCOM failed, error: ",
             OpenCLErrorToString(error));

  CreateQualcommBufferIONHostPtr(*pitch * shape[1], ion_host);
}
#endif  // MACE_ENABLE_RPCMEM
}  // namespace mace
