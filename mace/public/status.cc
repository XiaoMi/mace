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

#include <sstream>

#include "mace/public/mace.h"

namespace mace {

class MaceStatus::Impl {
 public:
  explicit Impl(const Code code): code_(code), information_("") {}
  Impl(const Code code, const std::string &informaton)
      : code_(code), information_(informaton) {}
  ~Impl() = default;

  void SetCode(const Code code) { code_ = code; }
  Code code() const { return code_; }
  void SetInformation(const std::string &info) { information_ = info; }
  std::string information() const {
    if (information_.empty()) {
      return CodeToString();
    } else {
      return CodeToString() + ": " + information_;
    }
  }

 private:
  std::string CodeToString() const {
    switch (code_) {
      case MaceStatus::MACE_SUCCESS:
        return "Success";
      case MaceStatus::MACE_INVALID_ARGS:
        return "Invalid Arguments";
      case MaceStatus::MACE_OUT_OF_RESOURCES:
        return "Out of resources";
      case MACE_UNSUPPORTED:
        return "Unsupported";
      case MACE_RUNTIME_ERROR:
        return "Runtime error";
      default:
        std::ostringstream os;
        os << code_;
        return os.str();
    }
  }

 private:
  MaceStatus::Code code_;
  std::string information_;
};

MaceStatus::MaceStatus()
    : impl_(new MaceStatus::Impl(MaceStatus::MACE_SUCCESS)) {}
MaceStatus::MaceStatus(const Code code) : impl_(new MaceStatus::Impl(code)) {}
MaceStatus::MaceStatus(const Code code, const std::string &information)
    : impl_(new MaceStatus::Impl(code, information)) {}
MaceStatus::MaceStatus(const MaceStatus &other)
    : impl_(new MaceStatus::Impl(other.code(), other.information())) {}
MaceStatus::MaceStatus(MaceStatus &&other)
    : impl_(new MaceStatus::Impl(other.code(), other.information())) {}
MaceStatus::~MaceStatus() = default;

MaceStatus& MaceStatus::operator=(const MaceStatus &other) {
  impl_->SetCode(other.code());
  impl_->SetInformation(other.information());
  return *this;
}
MaceStatus& MaceStatus::operator=(const MaceStatus &&other) {
  impl_->SetCode(other.code());
  impl_->SetInformation(other.information());
  return *this;
}

MaceStatus::Code MaceStatus::code() const {
  return impl_->code();
}

std::string MaceStatus::information() const {
  return impl_->information();
}

bool MaceStatus::operator==(const MaceStatus &other) const {
  return other.code() == impl_->code();
}

bool MaceStatus::operator!=(const MaceStatus &other) const {
  return other.code() != impl_->code();
}

}  // namespace mace
