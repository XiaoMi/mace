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


#include "mace/core/tensor.h"

namespace mace {

namespace numerical_chars {
std::ostream &operator<<(std::ostream &os, char c) {
  return std::is_signed<char>::value ? os << static_cast<int>(c)
                                     : os << static_cast<unsigned int>(c);
}

std::ostream &operator<<(std::ostream &os, signed char c) {
  return os << static_cast<int>(c);
}

std::ostream &operator<<(std::ostream &os, unsigned char c) {
  return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars

std::string Tensor::name() const {
  return name_;
}

DataType Tensor::dtype() const {
  return buffer_->data_type;
}

void Tensor::SetDtype(DataType dtype) {
  MACE_CHECK(dtype != DataType::DT_INVALID);
  buffer_->data_type = dtype;
}

bool Tensor::unused() const { return unused_; }

const std::vector<index_t> &Tensor::shape() const {
  return shape_;
}

std::vector<index_t> Tensor::max_shape() const {
  if (shape_configured_.empty()) {
    return shape();
  } else {
    auto &_shape = shape();
    std::vector<index_t> max_shape(_shape.size());
    MACE_CHECK(_shape.size() == shape_configured_.size());
    for (size_t i = 0; i < shape_configured_.size(); ++i) {
      max_shape[i] = std::max(_shape[i], shape_configured_[i]);
    }
    return max_shape;
  }
}

index_t Tensor::max_size() const {
  auto _max_shape = max_shape();
  return std::accumulate(_max_shape.begin(),
                         _max_shape.end(),
                         1,
                         std::multiplies<index_t>());
}

index_t Tensor::raw_max_size() const { return max_size() * SizeOfType(); }

void Tensor::SetShapeConfigured(const std::vector<index_t> &shape_configured) {
  shape_configured_ = shape_configured;
  shape_ = shape_configured;
}

void Tensor::SetContentType(BufferContentType content_type,
                            unsigned int content_param) {
  content_type_ = content_type;
  content_param_ = content_param;
}

void Tensor::GetContentType(BufferContentType *content_type,
                            unsigned int *content_param) const {
  MACE_CHECK(content_type != nullptr && content_param != nullptr);
  *content_type = content_type_;
  *content_param = content_param_;
}

const std::vector<index_t> &Tensor::buffer_shape() const {
  MACE_CHECK(buffer_ != nullptr);
  return buffer_->dims;
}

index_t Tensor::dim_size() const { return shape_.size(); }

index_t Tensor::dim(unsigned int index) const {
  MACE_CHECK(index < shape_.size(),
             name_, ": Dim out of range: ", index, " >= ", shape_.size());
  return shape_[index];
}

index_t Tensor::size() const {
  return std::accumulate(shape_.begin(), shape_.end(), 1,
                         std::multiplies<int64_t>());
}

index_t Tensor::raw_size() const { return size() * SizeOfType(); }

void Tensor::set_data_format(DataFormat data_format) {
  data_format_ = data_format;
}

MemoryType Tensor::memory_type() const {
  return buffer_->mem_type;
}

DataFormat Tensor::data_format() const {
  return data_format_;
}

index_t Tensor::buffer_offset() const {
  return buffer_->offset();
}

Runtime *Tensor::GetCurRuntime() const {
  return runtime_;
}

const void *Tensor::raw_data() const {
  MACE_CHECK(buffer_ != nullptr, "buffer is null");
  return buffer_->data<void>();
}

void *Tensor::raw_mutable_data() {
  MACE_CHECK_NOTNULL(buffer_);

  return buffer_->mutable_data<void>();
}

void Tensor::MarkUnused() {
  unused_ = true;
}

void Tensor::Clear() {
  if (buffer_ != nullptr) {
    memset(buffer_->mutable_data<void>(), 0, buffer_->bytes());
  }
}

void Tensor::Reshape(const std::vector<index_t> &shape) {
  shape_ = shape;
}

void Tensor::to4dim() {
  if (shape_.size() == 1) {
    shape_ = {1, 1, 1, shape_[0]};
  } else if (shape_.size() == 2) {
    shape_ = {1, 1, shape_[0], shape_[1]};
  } else if (shape_.size() == 3) {
    shape_ = {1, shape_[0], shape_[1], shape_[2]};
  } 
}

MaceStatus Tensor::Resize(const std::vector<index_t> &shape) {
  MaceStatus ret = MaceStatus::MACE_SUCCESS;
  bool need_new =
      (buffer_->memory<void>() == nullptr || buffer_->dims.size() == 0);
  if (!need_new) {
    need_new = !runtime_->CanReuseBuffer(buffer_.get(), shape, content_type_,
                                         content_param_);
  }

  shape_ = shape;
  if (need_new) {
    LOG(WARNING) << "Tensor::Resize, allocate private mem, name: " << name()
                 << ", new shape: " << MakeString(shape);
    ret = runtime_->AllocateBufferForTensor(this, RENT_PRIVATE);
  } else {
    auto buf_shape = runtime_->ComputeBufDimFromTensorDim(
        shape, buffer_->mem_type, content_type_, content_param_);
    ret = buffer_->Resize(buf_shape);
  }

  return ret;
}

// Make this tensor reuse other tensor's buffer.
// This tensor has the same dtype, shape and image_shape.
// It could be reshaped later (with image shape unchanged).
void Tensor::ReuseTensorBuffer(const Tensor &other) {
  runtime_ = other.runtime_;
  buffer_ = other.buffer_;
}

MaceStatus Tensor::ResizeLike(const Tensor &other) {
  return ResizeLike(&other);
}

MaceStatus Tensor::ResizeLike(const Tensor *other) {
  return Resize(other->shape());
}

void Tensor::CopyBytes(const void *src, size_t bytes) {
  MappingGuard guard(this);
  memcpy(buffer_->mutable_data<void>(), src, bytes);
}

void Tensor::Copy(const Tensor &other) {
  MACE_CHECK(memory_type() == CPU_BUFFER || other.memory_type() == CPU_BUFFER,
             "If there is no CPU buffer, you should use Opencl Transform."),
  SetDtype(other.dtype());
  ResizeLike(other);
  MappingGuard map_other(&other);
  CopyBytes(other.raw_data(), other.raw_size());
}

size_t Tensor::SizeOfType() const {
  size_t type_size = 0;
  MACE_RUN_WITH_TYPE_ENUM(dtype(), type_size = sizeof(T));
  return type_size;
}

Buffer *Tensor::UnderlyingBuffer() const { return buffer_.get(); }

void Tensor::DebugPrint() const {
  using namespace numerical_chars;  // NOLINT(build/namespaces)
  std::stringstream os;
  os << "Tensor " << name_ << " size: [";
  for (index_t i : shape_) {
    os << i << ", ";
  }
  os << "], content:\n";

  for (int i = 0; i < size(); ++i) {
    if (i != 0 && i % shape_.back() == 0) {
      os << "\n";
    }
    MACE_RUN_WITH_TYPE_ENUM(
        dtype(), (os << this->data<T>()[i] << ", "));
  }
  LOG(INFO) << os.str();
}

void Tensor::Map(bool wait_for_finish) const {
  runtime_->MapBuffer(buffer_.get(), wait_for_finish);
}

void Tensor::UnMap() const {
  runtime_->UnMapBuffer(buffer_.get());
}

Tensor::MappingGuard::MappingGuard(const Tensor *tensor,
                                   bool wait_for_finish)
    : tensor_(tensor) {
  if (tensor_ != nullptr) {
    tensor_->Map(wait_for_finish);
  }
}

Tensor::MappingGuard::MappingGuard(MappingGuard &&other) {
  tensor_ = other.tensor_;
  other.tensor_ = nullptr;
}

Tensor::MappingGuard::~MappingGuard() {
  if (tensor_ != nullptr) {
    tensor_->UnMap();
  }
}

bool Tensor::is_weight() const {
  return is_weight_;
}

float Tensor::scale() const {
  return scale_;
}

int32_t Tensor::zero_point() const {
  return zero_point_;
}

// hexagon now uses min/max instead of scale and zero
float Tensor::minval() const {
  return minval_;
}

float Tensor::maxval() const {
  return maxval_;
}

void Tensor::SetScale(float scale) {
  scale_ = scale;
}

void Tensor::SetZeroPoint(int32_t zero_point) {
  zero_point_ = zero_point;
}

void Tensor::SetIsWeight(bool is_weight) {
  is_weight_ = is_weight;
}

void Tensor::SetMinVal(float minval) {
  minval_ = minval;
}

void Tensor::SetMaxVal(float maxval) {
  maxval_ = maxval;
}

}  // namespace mace
