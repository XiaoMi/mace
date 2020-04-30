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

#include <numeric>

#include "mace/core/mace_tensor_impl.h"
#include "mace/utils/logging.h"

namespace mace {

MaceTensor::MaceTensor(const std::vector<int64_t> &shape,
                       std::shared_ptr<void> data, const DataFormat format,
                       const IDataType data_type, const MemoryType mem_type) {
  MACE_CHECK_NOTNULL(data.get());
  MACE_CHECK(format == DataFormat::NONE || format == DataFormat::NHWC
                 || format == DataFormat::NCHW || format == DataFormat::OIHW,
             "MACE only support NONE, NHWC, NCHW and OIHW "
             "formats of input now.");
  MACE_CHECK(data_type > IDT_INVALID && data_type < IDT_END,
             "Invalid data type");
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = shape;
  impl_->data = data;
  impl_->format = format;
  impl_->data_type = data_type;
  impl_->mem_type = mem_type;
  impl_->buffer_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>());
}

MaceTensor::MaceTensor() {
  impl_ = make_unique<MaceTensor::Impl>();
}

MaceTensor::MaceTensor(const MaceTensor &other) {
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->data_type = other.data_type();
  impl_->mem_type = other.memory_type();
  impl_->buffer_size = other.impl_->buffer_size;
}

MaceTensor::MaceTensor(const MaceTensor &&other) {
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->data_type = other.data_type();
  impl_->mem_type = other.memory_type();
  impl_->buffer_size = other.impl_->buffer_size;
}

MaceTensor &MaceTensor::operator=(const MaceTensor &other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->data_type = other.data_type();
  impl_->mem_type = other.memory_type();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

MaceTensor &MaceTensor::operator=(const MaceTensor &&other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->data_type = other.data_type();
  impl_->mem_type = other.memory_type();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

MaceTensor::~MaceTensor() = default;

const std::vector<int64_t> &MaceTensor::shape() const { return impl_->shape; }

const std::shared_ptr<float> MaceTensor::data() const {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<float> MaceTensor::data() {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<void> MaceTensor::raw_data() const {
  return impl_->data;
}

std::shared_ptr<void> MaceTensor::raw_mutable_data() {
  return impl_->data;
}

DataFormat MaceTensor::data_format() const {
  return impl_->format;
}

IDataType MaceTensor::data_type() const {
  return impl_->data_type;
}

MemoryType MaceTensor::memory_type() const {
  return impl_->mem_type;
}

}  // namespace mace
