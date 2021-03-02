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


#include "mace/public/mace.h"

#include "mace/runtimes/opencl/core/opencl_context.h"

namespace mace {


class GPUContextBuilder::Impl {
 public:
  Impl();
  void SetStoragePath(const std::string &path);

  void SetOpenCLBinaryPaths(const std::vector<std::string> &paths);

  void SetOpenCLBinary(const unsigned char *data, const size_t size);

  void SetOpenCLParameterPath(const std::string &path);

  void SetOpenCLParameter(const unsigned char *data, const size_t size);

  std::shared_ptr<OpenclContext> Finalize();

 public:
  std::string storage_path_;
  std::vector<std::string> opencl_binary_paths_;
  std::string opencl_parameter_path_;
  const unsigned char *opencl_binary_ptr_;
  size_t opencl_binary_size_;
  const unsigned char *opencl_parameter_ptr_;
  size_t opencl_parameter_size_;
};

GPUContextBuilder::Impl::Impl()
    : storage_path_(""), opencl_binary_paths_(0), opencl_parameter_path_(""),
      opencl_binary_ptr_(nullptr), opencl_binary_size_(0),
      opencl_parameter_ptr_(nullptr), opencl_parameter_size_(0) {}

void GPUContextBuilder::Impl::SetStoragePath(const std::string &path) {
  storage_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  opencl_binary_paths_ = paths;
}

void GPUContextBuilder::Impl::SetOpenCLBinary(const unsigned char *data,
                                              const size_t size) {
  opencl_binary_ptr_ = data;
  opencl_binary_size_ = size;
}

void GPUContextBuilder::Impl::SetOpenCLParameterPath(
    const std::string &path) {
  opencl_parameter_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLParameter(const unsigned char *data,
                                                 const size_t size) {
  opencl_parameter_ptr_ = data;
  opencl_parameter_size_ = size;
}


std::shared_ptr<OpenclContext> GPUContextBuilder::Impl::Finalize() {
#ifdef MACE_ENABLE_OPENCL
  return std::shared_ptr<OpenclContext>(new OpenclContext(
      storage_path_, opencl_binary_paths_, opencl_parameter_path_,
      opencl_binary_ptr_, opencl_binary_size_, opencl_parameter_ptr_,
      opencl_parameter_size_));
#else
  return nullptr;
#endif  // MACE_ENABLE_OPENCL
}


GPUContextBuilder::GPUContextBuilder() : impl_(new GPUContextBuilder::Impl) {}

GPUContextBuilder::~GPUContextBuilder() = default;

GPUContextBuilder &GPUContextBuilder::SetStoragePath(const std::string &path) {
  impl_->SetStoragePath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  impl_->SetOpenCLBinaryPaths(paths);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinary(
    const unsigned char *data, const size_t size) {
  impl_->SetOpenCLBinary(data, size);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameterPath(
    const std::string &path) {
  impl_->SetOpenCLParameterPath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameter(
    const unsigned char *data, const size_t size) {
  impl_->SetOpenCLParameter(data, size);
  return *this;
}

std::shared_ptr<OpenclContext> GPUContextBuilder::Finalize() {
  return impl_->Finalize();
}

}  // namespace mace
