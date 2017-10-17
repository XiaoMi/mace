//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "CL/opencl.h"

#include "mace/core/logging.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

#include <dlfcn.h>
#include <mutex>

/**
 * Wrapper of OpenCL 2.0 (based on 1.2)
 */
namespace mace {

namespace {
class OpenCLLibraryImpl final {
 public:
  static OpenCLLibraryImpl &Get();
  bool Load();
  void Unload();
  bool loaded() { return handle_ != nullptr; }

  using clGetPlatformIDsFunc = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
  using clGetPlatformInfoFunc =
      cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  using clBuildProgramFunc = cl_int (*)(cl_program,
                                        cl_uint,
                                        const cl_device_id *,
                                        const char *,
                                        void (*pfn_notify)(cl_program, void *),
                                        void *);
  using clEnqueueNDRangeKernelFunc = cl_int (*)(cl_command_queue,
                                                cl_kernel,
                                                cl_uint,
                                                const size_t *,
                                                const size_t *,
                                                const size_t *,
                                                cl_uint,
                                                const cl_event *,
                                                cl_event *);
  using clSetKernelArgFunc = cl_int (*)(cl_kernel,
                                        cl_uint,
                                        size_t,
                                        const void *);
  using clRetainMemObjectFunc = cl_int (*)(cl_mem);
  using clReleaseMemObjectFunc = cl_int (*)(cl_mem);
  using clEnqueueUnmapMemObjectFunc = cl_int (*)(
      cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  using clRetainCommandQueueFunc = cl_int (*)(cl_command_queue command_queue);
  using clCreateContextFunc = cl_context (*)(
      const cl_context_properties *,
      cl_uint,
      const cl_device_id *,
      void(CL_CALLBACK *)(const char *, const void *, size_t, void *),
      void *,
      cl_int *);
  using clCreateContextFromTypeFunc = cl_context (*)(
      const cl_context_properties *,
      cl_device_type,
      void(CL_CALLBACK *)(const char *, const void *, size_t, void *),
      void *,
      cl_int *);
  using clReleaseContextFunc = cl_int (*)(cl_context);
  using clWaitForEventsFunc = cl_int (*)(cl_uint, const cl_event *);
  using clReleaseEventFunc = cl_int (*)(cl_event);
  using clEnqueueWriteBufferFunc = cl_int (*)(cl_command_queue,
                                              cl_mem,
                                              cl_bool,
                                              size_t,
                                              size_t,
                                              const void *,
                                              cl_uint,
                                              const cl_event *,
                                              cl_event *);
  using clEnqueueReadBufferFunc = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_bool,
                                             size_t,
                                             size_t,
                                             void *,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
  using clGetProgramBuildInfoFunc = cl_int (*)(cl_program,
                                               cl_device_id,
                                               cl_program_build_info,
                                               size_t,
                                               void *,
                                               size_t *);
  using clRetainProgramFunc = cl_int (*)(cl_program program);
  using clEnqueueMapBufferFunc = void *(*)(cl_command_queue,
                                           cl_mem,
                                           cl_bool,
                                           cl_map_flags,
                                           size_t,
                                           size_t,
                                           cl_uint,
                                           const cl_event *,
                                           cl_event *,
                                           cl_int *);
  using clCreateCommandQueueWithPropertiesFunc =
      cl_command_queue (*)(cl_context /* context */,
                           cl_device_id /* device */,
                           const cl_queue_properties * /* properties */,
                           cl_int * /* errcode_ret */);
  using clReleaseCommandQueueFunc = cl_int (*)(cl_command_queue);
  using clCreateProgramWithBinaryFunc = cl_program (*)(cl_context,
                                                       cl_uint,
                                                       const cl_device_id *,
                                                       const size_t *,
                                                       const unsigned char **,
                                                       cl_int *,
                                                       cl_int *);
  using clRetainContextFunc = cl_int (*)(cl_context context);
  using clGetContextInfoFunc =
      cl_int (*)(cl_context, cl_context_info, size_t, void *, size_t *);
  using clReleaseProgramFunc = cl_int (*)(cl_program program);
  using clFlushFunc = cl_int (*)(cl_command_queue command_queue);
  using clFinishFunc = cl_int (*)(cl_command_queue command_queue);
  using clGetProgramInfoFunc =
      cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
  using clCreateKernelFunc = cl_kernel (*)(cl_program, const char *, cl_int *);
  using clRetainKernelFunc = cl_int (*)(cl_kernel kernel);
  using clCreateBufferFunc =
      cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
  using clCreateProgramWithSourceFunc = cl_program (*)(
      cl_context, cl_uint, const char **, const size_t *, cl_int *);
  using clReleaseKernelFunc = cl_int (*)(cl_kernel kernel);
  using clGetDeviceInfoFunc =
      cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
  using clGetDeviceIDsFunc = cl_int (*)(
      cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  using clRetainDeviceFunc = cl_int (*)(cl_device_id);
  using clReleaseDeviceFunc = cl_int (*)(cl_device_id);
  using clRetainEventFunc = cl_int (*)(cl_event);

#define DEFINE_FUNC_PTR(func) func##Func func = nullptr

  DEFINE_FUNC_PTR(clGetPlatformIDs);
  DEFINE_FUNC_PTR(clGetPlatformInfo);
  DEFINE_FUNC_PTR(clBuildProgram);
  DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
  DEFINE_FUNC_PTR(clSetKernelArg);
  DEFINE_FUNC_PTR(clReleaseKernel);
  DEFINE_FUNC_PTR(clCreateProgramWithSource);
  DEFINE_FUNC_PTR(clCreateBuffer);
  DEFINE_FUNC_PTR(clRetainKernel);
  DEFINE_FUNC_PTR(clCreateKernel);
  DEFINE_FUNC_PTR(clGetProgramInfo);
  DEFINE_FUNC_PTR(clFlush);
  DEFINE_FUNC_PTR(clFinish);
  DEFINE_FUNC_PTR(clReleaseProgram);
  DEFINE_FUNC_PTR(clRetainContext);
  DEFINE_FUNC_PTR(clGetContextInfo);
  DEFINE_FUNC_PTR(clCreateProgramWithBinary);
  DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
  DEFINE_FUNC_PTR(clReleaseCommandQueue);
  DEFINE_FUNC_PTR(clEnqueueMapBuffer);
  DEFINE_FUNC_PTR(clRetainProgram);
  DEFINE_FUNC_PTR(clGetProgramBuildInfo);
  DEFINE_FUNC_PTR(clEnqueueReadBuffer);
  DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
  DEFINE_FUNC_PTR(clWaitForEvents);
  DEFINE_FUNC_PTR(clReleaseEvent);
  DEFINE_FUNC_PTR(clCreateContext);
  DEFINE_FUNC_PTR(clCreateContextFromType);
  DEFINE_FUNC_PTR(clReleaseContext);
  DEFINE_FUNC_PTR(clRetainCommandQueue);
  DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
  DEFINE_FUNC_PTR(clRetainMemObject);
  DEFINE_FUNC_PTR(clReleaseMemObject);
  DEFINE_FUNC_PTR(clGetDeviceInfo);
  DEFINE_FUNC_PTR(clGetDeviceIDs);
  DEFINE_FUNC_PTR(clRetainDevice);
  DEFINE_FUNC_PTR(clReleaseDevice);
  DEFINE_FUNC_PTR(clRetainEvent);

#undef DEFINE_FUNC_PTR

 private:
  void *LoadFromPath(const std::string &path);
  void *handle_ = nullptr;
};

OpenCLLibraryImpl &OpenCLLibraryImpl::Get() {
  static std::once_flag load_once;
  static OpenCLLibraryImpl instance;
  std::call_once(load_once, []() { instance.Load(); });
  return instance;
}

bool OpenCLLibraryImpl::Load() {
  if (loaded()) return true;

  // TODO (heliangliang) Make this configurable
  // TODO (heliangliang) Benchmark 64 bit overhead
  static const std::vector<std::string> paths = {
#if defined(__aarch64__)
    // Qualcomm Adreno
    "/system/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    // Mali
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/lib64/egl/libGLES_mali.so",
#else
    // Qualcomm Adreno
    "/system/vendor/lib/libOpenCL.so",
    "/system/lib/libOpenCL.so",
    // Mali
    "/system/vendor/lib/egl/libGLES_mali.so",
    "/system/lib/egl/libGLES_mali.so",
#endif
  };

  for (const auto &path : paths) {
    VLOG(2) << "Loading OpenCL from " << path;
    void *handle = LoadFromPath(path);
    if (handle != nullptr) {
      handle_ = handle;
      return true;
    }
  }

  LOG(ERROR) << "Failed to load OpenCL library";
  return false;
}

void OpenCLLibraryImpl::Unload() {
  if (handle_ != nullptr) {
    if (dlclose(handle_) != 0) {
      LOG(ERROR) << "dlclose failed for OpenCL library";
    }
    handle_ = nullptr;
  }
}

void *OpenCLLibraryImpl::LoadFromPath(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (handle == nullptr) {
    VLOG(2) << "Failed to load OpenCL library from path " << path
            << " error code: " << dlerror();
    return nullptr;
  }

#define ASSIGN_FROM_DLSYM(func)                                     \
  do {                                                              \
    void *ptr = dlsym(handle, #func);                               \
    if (ptr == nullptr) {                                           \
      LOG(ERROR) << "Failed to load " << #func << " from " << path; \
      dlclose(handle);                                              \
      return nullptr;                                               \
    }                                                               \
    func = reinterpret_cast<func##Func>(ptr);                       \
    VLOG(2) << "Loaded " << #func << " from " << path;              \
  } while (false)

  ASSIGN_FROM_DLSYM(clGetPlatformIDs);
  ASSIGN_FROM_DLSYM(clGetPlatformInfo);
  ASSIGN_FROM_DLSYM(clBuildProgram);
  ASSIGN_FROM_DLSYM(clEnqueueNDRangeKernel);
  ASSIGN_FROM_DLSYM(clSetKernelArg);
  ASSIGN_FROM_DLSYM(clReleaseKernel);
  ASSIGN_FROM_DLSYM(clCreateProgramWithSource);
  ASSIGN_FROM_DLSYM(clCreateBuffer);
  ASSIGN_FROM_DLSYM(clRetainKernel);
  ASSIGN_FROM_DLSYM(clCreateKernel);
  ASSIGN_FROM_DLSYM(clGetProgramInfo);
  ASSIGN_FROM_DLSYM(clFlush);
  ASSIGN_FROM_DLSYM(clFinish);
  ASSIGN_FROM_DLSYM(clReleaseProgram);
  ASSIGN_FROM_DLSYM(clRetainContext);
  ASSIGN_FROM_DLSYM(clGetContextInfo);
  ASSIGN_FROM_DLSYM(clCreateProgramWithBinary);
  ASSIGN_FROM_DLSYM(clCreateCommandQueueWithProperties);
  ASSIGN_FROM_DLSYM(clReleaseCommandQueue);
  ASSIGN_FROM_DLSYM(clEnqueueMapBuffer);
  ASSIGN_FROM_DLSYM(clRetainProgram);
  ASSIGN_FROM_DLSYM(clGetProgramBuildInfo);
  ASSIGN_FROM_DLSYM(clEnqueueReadBuffer);
  ASSIGN_FROM_DLSYM(clEnqueueWriteBuffer);
  ASSIGN_FROM_DLSYM(clWaitForEvents);
  ASSIGN_FROM_DLSYM(clReleaseEvent);
  ASSIGN_FROM_DLSYM(clCreateContext);
  ASSIGN_FROM_DLSYM(clCreateContextFromType);
  ASSIGN_FROM_DLSYM(clReleaseContext);
  ASSIGN_FROM_DLSYM(clRetainCommandQueue);
  ASSIGN_FROM_DLSYM(clEnqueueUnmapMemObject);
  ASSIGN_FROM_DLSYM(clRetainMemObject);
  ASSIGN_FROM_DLSYM(clReleaseMemObject);
  ASSIGN_FROM_DLSYM(clGetDeviceInfo);
  ASSIGN_FROM_DLSYM(clGetDeviceIDs);
  ASSIGN_FROM_DLSYM(clRetainDevice);
  ASSIGN_FROM_DLSYM(clReleaseDevice);
  ASSIGN_FROM_DLSYM(clRetainEvent);

#undef ASSIGN_FROM_DLSYM

  return handle;
}
}  // namespace

bool OpenCLLibrary::Supported() { return OpenCLLibraryImpl::Get().loaded(); }

void OpenCLLibrary::Load() { OpenCLLibraryImpl::Get().Load(); }

void OpenCLLibrary::Unload() { OpenCLLibraryImpl::Get().Unload(); }

}  // namespace mace

cl_int clGetPlatformIDs(cl_uint num_entries,
                        cl_platform_id *platforms,
                        cl_uint *num_platforms) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetPlatformIDs;
  if (func != nullptr) {
    return func(num_entries, platforms, num_platforms);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}
cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetPlatformInfo;
  if (func != nullptr) {
    return func(platform, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clBuildProgram(cl_program program,
                      cl_uint num_devices,
                      const cl_device_id *device_list,
                      const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                    void *user_data),
                      void *user_data) {
  auto func = mace::OpenCLLibraryImpl::Get().clBuildProgram;
  if (func != nullptr) {
    return func(program, num_devices, device_list, options, pfn_notify,
                user_data);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                              cl_kernel kernel,
                              cl_uint work_dim,
                              const size_t *global_work_offset,
                              const size_t *global_work_size,
                              const size_t *local_work_size,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event) {
  auto func = mace::OpenCLLibraryImpl::Get().clEnqueueNDRangeKernel;
  if (func != nullptr) {
    return func(command_queue, kernel, work_dim, global_work_offset,
                global_work_size, local_work_size, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clSetKernelArg(cl_kernel kernel,
                      cl_uint arg_index,
                      size_t arg_size,
                      const void *arg_value) {
  auto func = mace::OpenCLLibraryImpl::Get().clSetKernelArg;
  if (func != nullptr) {
    return func(kernel, arg_index, arg_size, arg_value);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clRetainMemObject(cl_mem memobj) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainMemObject;
  if (func != nullptr) {
    return func(memobj);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clReleaseMemObject(cl_mem memobj) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseMemObject;
  if (func != nullptr) {
    return func(memobj);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                               cl_mem memobj,
                               void *mapped_ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event) {
  auto func = mace::OpenCLLibraryImpl::Get().clEnqueueUnmapMemObject;
  if (func != nullptr) {
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainCommandQueue;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}
cl_context clCreateContext(
    const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateContext;
  if (func != nullptr) {
    return func(properties, num_devices, devices, pfn_notify, user_data,
                errcode_ret);
  } else {
    return nullptr;
  }
}
cl_context clCreateContextFromType(
    const cl_context_properties *properties,
    cl_device_type device_type,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateContextFromType;
  if (func != nullptr) {
    return func(properties, device_type, pfn_notify, user_data, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseContext(cl_context context) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseContext;
  if (func != nullptr) {
    return func(context);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  auto func = mace::OpenCLLibraryImpl::Get().clWaitForEvents;
  if (func != nullptr) {
    return func(num_events, event_list);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clReleaseEvent(cl_event event) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseEvent;
  if (func != nullptr) {
    return func(event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                            cl_mem buffer,
                            cl_bool blocking_write,
                            size_t offset,
                            size_t size,
                            const void *ptr,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event) {
  auto func = mace::OpenCLLibraryImpl::Get().clEnqueueWriteBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_write, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                           cl_mem buffer,
                           cl_bool blocking_read,
                           size_t offset,
                           size_t size,
                           void *ptr,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event) {
  auto func = mace::OpenCLLibraryImpl::Get().clEnqueueReadBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_read, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clGetProgramBuildInfo(cl_program program,
                             cl_device_id device,
                             cl_program_build_info param_name,
                             size_t param_value_size,
                             void *param_value,
                             size_t *param_value_size_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetProgramBuildInfo;
  if (func != nullptr) {
    return func(program, device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clRetainProgram(cl_program program) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainProgram;
  if (func != nullptr) {
    return func(program);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

void *clEnqueueMapBuffer(cl_command_queue command_queue,
                         cl_mem buffer,
                         cl_bool blocking_map,
                         cl_map_flags map_flags,
                         size_t offset,
                         size_t size,
                         cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clEnqueueMapBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_map, map_flags, offset, size,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
  } else {
    if (errcode_ret != nullptr) {
      *errcode_ret = CL_OUT_OF_RESOURCES;
    }
    return nullptr;
  }
}
cl_command_queue clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties *properties,
    cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateCommandQueueWithProperties;
  if (func != nullptr) {
    return func(context, device, properties, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseCommandQueue;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_program clCreateProgramWithBinary(cl_context context,
                                     cl_uint num_devices,
                                     const cl_device_id *device_list,
                                     const size_t *lengths,
                                     const unsigned char **binaries,
                                     cl_int *binary_status,
                                     cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateProgramWithBinary;
  if (func != nullptr) {
    return func(context, num_devices, device_list, lengths, binaries,
                binary_status, errcode_ret);
  } else {
    if (errcode_ret != nullptr) {
      *errcode_ret = CL_OUT_OF_RESOURCES;
    }
    return nullptr;
  }
}

cl_int clRetainContext(cl_context context) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainContext;
  if (func != nullptr) {
    return func(context);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clGetContextInfo(cl_context context,
                        cl_context_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetContextInfo;
  if (func != nullptr) {
    return func(context, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clReleaseProgram(cl_program program) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseProgram;
  if (func != nullptr) {
    return func(program);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clFlush(cl_command_queue command_queue) {
  auto func = mace::OpenCLLibraryImpl::Get().clFlush;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clFinish(cl_command_queue command_queue) {
  auto func = mace::OpenCLLibraryImpl::Get().clFinish;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clGetProgramInfo(cl_program program,
                        cl_program_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetProgramInfo;
  if (func != nullptr) {
    return func(program, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_kernel clCreateKernel(cl_program program,
                         const char *kernel_name,
                         cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateKernel;
  if (func != nullptr) {
    return func(program, kernel_name, errcode_ret);
  } else {
    if (errcode_ret != nullptr) {
      *errcode_ret = CL_OUT_OF_RESOURCES;
    }
    return nullptr;
  }
}

cl_int clRetainKernel(cl_kernel kernel) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainKernel;
  if (func != nullptr) {
    return func(kernel);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_mem clCreateBuffer(cl_context context,
                      cl_mem_flags flags,
                      size_t size,
                      void *host_ptr,
                      cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateBuffer;
  if (func != nullptr) {
    return func(context, flags, size, host_ptr, errcode_ret);
  } else {
    if (errcode_ret != nullptr) {
      *errcode_ret = CL_OUT_OF_RESOURCES;
    }
    return nullptr;
  }
}

cl_program clCreateProgramWithSource(cl_context context,
                                     cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clCreateProgramWithSource;
  if (func != nullptr) {
    return func(context, count, strings, lengths, errcode_ret);
  } else {
    if (errcode_ret != nullptr) {
      *errcode_ret = CL_OUT_OF_RESOURCES;
    }
    return nullptr;
  }
}

cl_int clReleaseKernel(cl_kernel kernel) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseKernel;
  if (func != nullptr) {
    return func(kernel);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clGetDeviceIDs(cl_platform_id platform,
                      cl_device_type device_type,
                      cl_uint num_entries,
                      cl_device_id *devices,
                      cl_uint *num_devices) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetDeviceIDs;
  if (func != nullptr) {
    return func(platform, device_type, num_entries, devices, num_devices);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clGetDeviceInfo(cl_device_id device,
                       cl_device_info param_name,
                       size_t param_value_size,
                       void *param_value,
                       size_t *param_value_size_ret) {
  auto func = mace::OpenCLLibraryImpl::Get().clGetDeviceInfo;
  if (func != nullptr) {
    return func(device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clRetainDevice(cl_device_id device) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainDevice;
  if (func != nullptr) {
    return func(device);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clReleaseDevice(cl_device_id device) {
  auto func = mace::OpenCLLibraryImpl::Get().clReleaseDevice;
  if (func != nullptr) {
    return func(device);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}

cl_int clRetainEvent(cl_event event) {
  auto func = mace::OpenCLLibraryImpl::Get().clRetainEvent;
  if (func != nullptr) {
    return func(event);
  } else {
    return CL_OUT_OF_RESOURCES;
  }
}
