//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/cl2_header.h"
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
    case DT_INT8:
    case DT_INT16:
    case DT_INT32:
      return CL_SIGNED_INT32;
    case DT_UINT8:
    case DT_UINT16:
    case DT_UINT32:
      return CL_UNSIGNED_INT32;
    default:
      LOG(FATAL) << "Image doesn't support the data type: " << t;
      return 0;
  }
}

}

OpenCLAllocator::OpenCLAllocator() {}

OpenCLAllocator::~OpenCLAllocator() {}
void *OpenCLAllocator::New(size_t nbytes) {
  VLOG(3) << "Allocate OpenCL buffer: " << nbytes;
  cl_int error;
  cl::Buffer *buffer = new cl::Buffer(OpenCLRuntime::Global()->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      nbytes, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS);
  MACE_CHECK_NOTNULL(buffer);
  return static_cast<void *>(buffer);
}

void *OpenCLAllocator::NewImage(const std::vector<size_t> &image_shape,
                                const DataType dt) {
  MACE_CHECK(image_shape.size() == 2) << "Image shape's size must equal 2";
  VLOG(3) << "Allocate OpenCL image: " << image_shape[0] << ", " << image_shape[1];

  cl::ImageFormat img_format(CL_RGBA, DataTypeToCLChannelType(dt));

  cl_int error;
  cl::Image2D *cl_image =
      new cl::Image2D(OpenCLRuntime::Global()->context(),
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                      img_format,
                      image_shape[0], image_shape[1],
                      0, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS) << error << " with image shape: ["
                                  << image_shape[0] << ", " << image_shape[1]
                                  << "]";

  return cl_image;
}

void OpenCLAllocator::Delete(void *buffer) {
  VLOG(3) << "Free OpenCL buffer";
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

void OpenCLAllocator::DeleteImage(void *buffer) {
  VLOG(3) << "Free OpenCL image";
  if (buffer != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(buffer);
    delete cl_image;
  }
}

void *OpenCLAllocator::Map(void *buffer, size_t nbytes) {
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = OpenCLRuntime::Global()->command_queue();
  // TODO(heliangliang) Non-blocking call
  cl_int error;
  void *mapped_ptr =
      queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                             nbytes, nullptr, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS);
  return mapped_ptr;
}

// TODO : there is something wrong with half type.
void *OpenCLAllocator::MapImage(void *buffer,
                                const std::vector<size_t> &image_shape,
                                std::vector<size_t> &mapped_image_pitch) {
  MACE_CHECK(image_shape.size() == 2) << "Just support map 2d image";
  auto cl_image = static_cast<cl::Image2D *>(buffer);
  std::array<size_t, 3> origin = {0, 0, 0};
  std::array<size_t, 3> region = {image_shape[0], image_shape[1], 1};

  mapped_image_pitch.resize(2);
  cl_int error;
  void *mapped_ptr =
      OpenCLRuntime::Global()->command_queue().enqueueMapImage(*cl_image,
                                                            CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                                            origin, region,
                                                            &mapped_image_pitch[0],
                                                            &mapped_image_pitch[1],
                                                            nullptr, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS) << error;

  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer, void *mapped_ptr) {
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = OpenCLRuntime::Global()->command_queue();
  MACE_CHECK(queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr, nullptr,
                                         nullptr) == CL_SUCCESS);
}

bool OpenCLAllocator::OnHost() { return false; }

MACE_REGISTER_ALLOCATOR(DeviceType::OPENCL, new OpenCLAllocator());

}  // namespace mace
