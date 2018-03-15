//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_wrapper.h"

#include <CL/opencl.h>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "mace/utils/logging.h"

/**
 * Wrapper of OpenCL 2.0 (based on 1.2)
 */
namespace mace {

namespace {
class OpenCLLibraryImpl final {
 public:
  bool Load();
  void Unload();

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
      void (CL_CALLBACK *)(const char *, const void *, size_t, void *),  // NOLINT
      void *,
      cl_int *);
  using clCreateContextFromTypeFunc = cl_context (*)(
      const cl_context_properties *,
      cl_device_type,
      void (CL_CALLBACK *)(const char *, const void *, size_t, void *),  // NOLINT
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
  using clEnqueueMapImageFunc = void *(*)(cl_command_queue,
                                          cl_mem,
                                          cl_bool,
                                          cl_map_flags,
                                          const size_t *,
                                          const size_t *,
                                          size_t *,
                                          size_t *,
                                          cl_uint,
                                          const cl_event *,
                                          cl_event *,
                                          cl_int *);
  using clCreateCommandQueueWithPropertiesFunc = cl_command_queue (*)(
      cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
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
  using clCreateImageFunc = cl_mem (*)(cl_context,
                                       cl_mem_flags,
                                       const cl_image_format *,
                                       const cl_image_desc *,
                                       void *,
                                       cl_int *);
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
  using clGetKernelWorkGroupInfoFunc = cl_int (*)(cl_kernel,
                                                  cl_device_id,
                                                  cl_kernel_work_group_info,
                                                  size_t,
                                                  void *,
                                                  size_t *);
  using clGetEventProfilingInfoFunc = cl_int (*)(cl_event event,
                                                 cl_profiling_info param_name,
                                                 size_t param_value_size,
                                                 void *param_value,
                                                 size_t *param_value_size_ret);
  using clGetImageInfoFunc =
      cl_int (*)(cl_mem, cl_image_info, size_t, void *, size_t *);

#define MACE_CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr

  MACE_CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
  MACE_CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
  MACE_CL_DEFINE_FUNC_PTR(clBuildProgram);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
  MACE_CL_DEFINE_FUNC_PTR(clSetKernelArg);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseKernel);
  MACE_CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
  MACE_CL_DEFINE_FUNC_PTR(clCreateBuffer);
  MACE_CL_DEFINE_FUNC_PTR(clCreateImage);
  MACE_CL_DEFINE_FUNC_PTR(clRetainKernel);
  MACE_CL_DEFINE_FUNC_PTR(clCreateKernel);
  MACE_CL_DEFINE_FUNC_PTR(clGetProgramInfo);
  MACE_CL_DEFINE_FUNC_PTR(clFlush);
  MACE_CL_DEFINE_FUNC_PTR(clFinish);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseProgram);
  MACE_CL_DEFINE_FUNC_PTR(clRetainContext);
  MACE_CL_DEFINE_FUNC_PTR(clGetContextInfo);
  MACE_CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);
  MACE_CL_DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueMapImage);
  MACE_CL_DEFINE_FUNC_PTR(clRetainProgram);
  MACE_CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
  MACE_CL_DEFINE_FUNC_PTR(clWaitForEvents);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseEvent);
  MACE_CL_DEFINE_FUNC_PTR(clCreateContext);
  MACE_CL_DEFINE_FUNC_PTR(clCreateContextFromType);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseContext);
  MACE_CL_DEFINE_FUNC_PTR(clRetainCommandQueue);
  MACE_CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
  MACE_CL_DEFINE_FUNC_PTR(clRetainMemObject);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseMemObject);
  MACE_CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
  MACE_CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
  MACE_CL_DEFINE_FUNC_PTR(clRetainDevice);
  MACE_CL_DEFINE_FUNC_PTR(clReleaseDevice);
  MACE_CL_DEFINE_FUNC_PTR(clRetainEvent);
  MACE_CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
  MACE_CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
  MACE_CL_DEFINE_FUNC_PTR(clGetImageInfo);

#undef MACE_CL_DEFINE_FUNC_PTR

 private:
  void *LoadFromPath(const std::string &path);
  void *handle_ = nullptr;
};

bool OpenCLLibraryImpl::Load() {
  if (handle_ != nullptr) {
    return true;
  }

  const std::vector<std::string> paths = {
    "libOpenCL.so",
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
      break;
    }
  }

  if (handle_ == nullptr) {
    LOG(ERROR) << "Failed to load OpenCL library";
    return false;
  }

  return true;
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

#define MACE_CL_ASSIGN_FROM_DLSYM(func)                             \
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

  MACE_CL_ASSIGN_FROM_DLSYM(clGetPlatformIDs);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetPlatformInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clBuildProgram);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueNDRangeKernel);
  MACE_CL_ASSIGN_FROM_DLSYM(clSetKernelArg);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseKernel);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateProgramWithSource);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateBuffer);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateImage);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainKernel);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateKernel);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetProgramInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clFlush);
  MACE_CL_ASSIGN_FROM_DLSYM(clFinish);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseProgram);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainContext);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetContextInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateProgramWithBinary);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateCommandQueueWithProperties);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseCommandQueue);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueMapBuffer);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueMapImage);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainProgram);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetProgramBuildInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueReadBuffer);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueWriteBuffer);
  MACE_CL_ASSIGN_FROM_DLSYM(clWaitForEvents);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseEvent);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateContext);
  MACE_CL_ASSIGN_FROM_DLSYM(clCreateContextFromType);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseContext);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainCommandQueue);
  MACE_CL_ASSIGN_FROM_DLSYM(clEnqueueUnmapMemObject);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainMemObject);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseMemObject);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetDeviceInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetDeviceIDs);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainDevice);
  MACE_CL_ASSIGN_FROM_DLSYM(clReleaseDevice);
  MACE_CL_ASSIGN_FROM_DLSYM(clRetainEvent);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetKernelWorkGroupInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetEventProfilingInfo);
  MACE_CL_ASSIGN_FROM_DLSYM(clGetImageInfo);

#undef MACE_CL_ASSIGN_FROM_DLSYM

  return handle;
}

OpenCLLibraryImpl *openclLibraryImpl = nullptr;
}  // namespace

void LoadOpenCLLibrary() {
  MACE_CHECK(openclLibraryImpl == nullptr);
  openclLibraryImpl = new OpenCLLibraryImpl();
  MACE_CHECK(openclLibraryImpl->Load());
}

void UnloadOpenCLLibrary() {
  MACE_CHECK_NOTNULL(openclLibraryImpl);
  openclLibraryImpl->Unload();
  delete openclLibraryImpl;
  openclLibraryImpl = nullptr;
}

}  // namespace mace

cl_int clGetPlatformIDs(cl_uint num_entries,
                        cl_platform_id *platforms,
                        cl_uint *num_platforms) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetPlatformIDs");
  auto func = mace::openclLibraryImpl->clGetPlatformIDs;
  MACE_CHECK_NOTNULL(func);
  return func(num_entries, platforms, num_platforms);
}
cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetPlatformInfo");
  auto func = mace::openclLibraryImpl->clGetPlatformInfo;
  MACE_CHECK_NOTNULL(func);
  return func(platform, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clBuildProgram(cl_program program,
                      cl_uint num_devices,
                      const cl_device_id *device_list,
                      const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                    void *user_data),
                      void *user_data) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clBuildProgram");
  auto func = mace::openclLibraryImpl->clBuildProgram;
  MACE_CHECK_NOTNULL(func);
  return func(program, num_devices, device_list, options, pfn_notify,
              user_data);
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
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueNDRangeKernel");
  auto func = mace::openclLibraryImpl->clEnqueueNDRangeKernel;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, kernel, work_dim, global_work_offset,
              global_work_size, local_work_size, num_events_in_wait_list,
              event_wait_list, event);
}

cl_int clSetKernelArg(cl_kernel kernel,
                      cl_uint arg_index,
                      size_t arg_size,
                      const void *arg_value) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clSetKernelArg");
  auto func = mace::openclLibraryImpl->clSetKernelArg;
  MACE_CHECK_NOTNULL(func);
  return func(kernel, arg_index, arg_size, arg_value);
}

cl_int clRetainMemObject(cl_mem memobj) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainMemObject");
  auto func = mace::openclLibraryImpl->clRetainMemObject;
  MACE_CHECK_NOTNULL(func);
  return func(memobj);
}

cl_int clReleaseMemObject(cl_mem memobj) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseMemObject");
  auto func = mace::openclLibraryImpl->clReleaseMemObject;
  MACE_CHECK_NOTNULL(func);
  return func(memobj);
}

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                               cl_mem memobj,
                               void *mapped_ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueUnmapMemObject");
  auto func = mace::openclLibraryImpl->clEnqueueUnmapMemObject;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list,
              event_wait_list, event);
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainCommandQueue");
  auto func = mace::openclLibraryImpl->clRetainCommandQueue;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue);
}

cl_context clCreateContext(
    const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateContext");
  auto func = mace::openclLibraryImpl->clCreateContext;
  MACE_CHECK_NOTNULL(func);
  return func(properties, num_devices, devices, pfn_notify, user_data,
              errcode_ret);
}

cl_context clCreateContextFromType(
    const cl_context_properties *properties,
    cl_device_type device_type,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateContextFromType");
  auto func = mace::openclLibraryImpl->clCreateContextFromType;
  MACE_CHECK_NOTNULL(func);
  return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

cl_int clReleaseContext(cl_context context) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseContext");
  auto func = mace::openclLibraryImpl->clReleaseContext;
  MACE_CHECK_NOTNULL(func);
  return func(context);
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clWaitForEvents");
  auto func = mace::openclLibraryImpl->clWaitForEvents;
  MACE_CHECK_NOTNULL(func);
  return func(num_events, event_list);
}

cl_int clReleaseEvent(cl_event event) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseEvent");
  auto func = mace::openclLibraryImpl->clReleaseEvent;
  MACE_CHECK_NOTNULL(func);
  return func(event);
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
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueWriteBuffer");
  auto func = mace::openclLibraryImpl->clEnqueueWriteBuffer;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, buffer, blocking_write, offset, size, ptr,
              num_events_in_wait_list, event_wait_list, event);
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
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueReadBuffer");
  auto func = mace::openclLibraryImpl->clEnqueueReadBuffer;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, buffer, blocking_read, offset, size, ptr,
              num_events_in_wait_list, event_wait_list, event);
}

cl_int clGetProgramBuildInfo(cl_program program,
                             cl_device_id device,
                             cl_program_build_info param_name,
                             size_t param_value_size,
                             void *param_value,
                             size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetProgramBuildInfo");
  auto func = mace::openclLibraryImpl->clGetProgramBuildInfo;
  MACE_CHECK_NOTNULL(func);
  return func(program, device, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clRetainProgram(cl_program program) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainProgram");
  auto func = mace::openclLibraryImpl->clRetainProgram;
  MACE_CHECK_NOTNULL(func);
  return func(program);
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
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueMapBuffer");
  auto func = mace::openclLibraryImpl->clEnqueueMapBuffer;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, buffer, blocking_map, map_flags, offset, size,
              num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

void *clEnqueueMapImage(cl_command_queue command_queue,
                        cl_mem image,
                        cl_bool blocking_map,
                        cl_map_flags map_flags,
                        const size_t origin[3],
                        const size_t region[3],
                        size_t *image_row_pitch,
                        size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event *event,
                        cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clEnqueueMapImage");
  auto func = mace::openclLibraryImpl->clEnqueueMapImage;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue, image, blocking_map, map_flags, origin, region,
              image_row_pitch, image_slice_pitch, num_events_in_wait_list,
              event_wait_list, event, errcode_ret);
}

cl_command_queue clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties *properties,
    cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateCommandQueueWithProperties");
  auto func = mace::openclLibraryImpl->clCreateCommandQueueWithProperties;
  MACE_CHECK_NOTNULL(func);
  return func(context, device, properties, errcode_ret);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseCommandQueue");
  auto func = mace::openclLibraryImpl->clReleaseCommandQueue;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue);
}

cl_program clCreateProgramWithBinary(cl_context context,
                                     cl_uint num_devices,
                                     const cl_device_id *device_list,
                                     const size_t *lengths,
                                     const unsigned char **binaries,
                                     cl_int *binary_status,
                                     cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateProgramWithBinary");
  auto func = mace::openclLibraryImpl->clCreateProgramWithBinary;
  MACE_CHECK_NOTNULL(func);
  return func(context, num_devices, device_list, lengths, binaries,
              binary_status, errcode_ret);
}

cl_int clRetainContext(cl_context context) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainContext");
  auto func = mace::openclLibraryImpl->clRetainContext;
  MACE_CHECK_NOTNULL(func);
  return func(context);
}

cl_int clGetContextInfo(cl_context context,
                        cl_context_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetContextInfo");
  auto func = mace::openclLibraryImpl->clGetContextInfo;
  MACE_CHECK_NOTNULL(func);
  return func(context, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clReleaseProgram(cl_program program) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseProgram");
  auto func = mace::openclLibraryImpl->clReleaseProgram;
  MACE_CHECK_NOTNULL(func);
  return func(program);
}

cl_int clFlush(cl_command_queue command_queue) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clFlush");
  auto func = mace::openclLibraryImpl->clFlush;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue);
}

cl_int clFinish(cl_command_queue command_queue) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clFinish");
  auto func = mace::openclLibraryImpl->clFinish;
  MACE_CHECK_NOTNULL(func);
  return func(command_queue);
}

cl_int clGetProgramInfo(cl_program program,
                        cl_program_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetProgramInfo");
  auto func = mace::openclLibraryImpl->clGetProgramInfo;
  MACE_CHECK_NOTNULL(func);
  return func(program, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_kernel clCreateKernel(cl_program program,
                         const char *kernel_name,
                         cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateKernel");
  auto func = mace::openclLibraryImpl->clCreateKernel;
  MACE_CHECK_NOTNULL(func);
  return func(program, kernel_name, errcode_ret);
}

cl_int clRetainKernel(cl_kernel kernel) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainKernel");
  auto func = mace::openclLibraryImpl->clRetainKernel;
  MACE_CHECK_NOTNULL(func);
  return func(kernel);
}

cl_mem clCreateBuffer(cl_context context,
                      cl_mem_flags flags,
                      size_t size,
                      void *host_ptr,
                      cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateBuffer");
  auto func = mace::openclLibraryImpl->clCreateBuffer;
  MACE_CHECK_NOTNULL(func);
  return func(context, flags, size, host_ptr, errcode_ret);
}

cl_mem clCreateImage(cl_context context,
                     cl_mem_flags flags,
                     const cl_image_format *image_format,
                     const cl_image_desc *image_desc,
                     void *host_ptr,
                     cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateImage");
  auto func = mace::openclLibraryImpl->clCreateImage;
  MACE_CHECK_NOTNULL(func);
  return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_program clCreateProgramWithSource(cl_context context,
                                     cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clCreateProgramWithSource");
  auto func = mace::openclLibraryImpl->clCreateProgramWithSource;
  MACE_CHECK_NOTNULL(func);
  return func(context, count, strings, lengths, errcode_ret);
}

cl_int clReleaseKernel(cl_kernel kernel) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseKernel");
  auto func = mace::openclLibraryImpl->clReleaseKernel;
  MACE_CHECK_NOTNULL(func);
  return func(kernel);
}

cl_int clGetDeviceIDs(cl_platform_id platform,
                      cl_device_type device_type,
                      cl_uint num_entries,
                      cl_device_id *devices,
                      cl_uint *num_devices) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetDeviceIDs");
  auto func = mace::openclLibraryImpl->clGetDeviceIDs;
  MACE_CHECK_NOTNULL(func);
  return func(platform, device_type, num_entries, devices, num_devices);
}

cl_int clGetDeviceInfo(cl_device_id device,
                       cl_device_info param_name,
                       size_t param_value_size,
                       void *param_value,
                       size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetDeviceInfo");
  auto func = mace::openclLibraryImpl->clGetDeviceInfo;
  MACE_CHECK_NOTNULL(func);
  return func(device, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clRetainDevice(cl_device_id device) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainDevice");
  auto func = mace::openclLibraryImpl->clRetainDevice;
  MACE_CHECK_NOTNULL(func);
  return func(device);
}

cl_int clReleaseDevice(cl_device_id device) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clReleaseDevice");
  auto func = mace::openclLibraryImpl->clReleaseDevice;
  MACE_CHECK_NOTNULL(func);
  return func(device);
}

cl_int clRetainEvent(cl_event event) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clRetainEvent");
  auto func = mace::openclLibraryImpl->clRetainEvent;
  MACE_CHECK_NOTNULL(func);
  return func(event);
}

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
                                cl_device_id device,
                                cl_kernel_work_group_info param_name,
                                size_t param_value_size,
                                void *param_value,
                                size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetKernelWorkGroupInfo");
  auto func = mace::openclLibraryImpl->clGetKernelWorkGroupInfo;
  MACE_CHECK_NOTNULL(func);
  return func(kernel, device, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clGetEventProfilingInfo(cl_event event,
                               cl_profiling_info param_name,
                               size_t param_value_size,
                               void *param_value,
                               size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetEventProfilingInfo");
  auto func = mace::openclLibraryImpl->clGetEventProfilingInfo;
  MACE_CHECK_NOTNULL(func);
  return func(event, param_name, param_value_size, param_value,
              param_value_size_ret);
}

cl_int clGetImageInfo(cl_mem image,
                      cl_image_info param_name,
                      size_t param_value_size,
                      void *param_value,
                      size_t *param_value_size_ret) {
  MACE_CHECK_NOTNULL(mace::openclLibraryImpl);
  MACE_LATENCY_LOGGER(3, "clGetImageInfo");
  auto func = mace::openclLibraryImpl->clGetImageInfo;
  MACE_CHECK_NOTNULL(func);
  return func(image, param_name, param_value_size, param_value,
              param_value_size_ret);
}
