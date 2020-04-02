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


#include "rpc/stub/base_handle.h"

namespace rpc {
namespace stub {

namespace {
const remote_handle64 IVALID_HANDLE = -1;
}

BaseHandle::BaseHandle(FuncOpen *func_open,
                       FuncClose *func_close,
                       const char *uri)
    : func_open_(func_open),
      func_close_(func_close),
      uri_(uri),
      remote_handle_(IVALID_HANDLE) {}

BaseHandle::~BaseHandle() {
  Close();
}

bool BaseHandle::Open() {
  if (Valid()) {
    return true;
  }

  int ret = func_open_(uri_, &remote_handle_);
  if (ret != 0 || remote_handle_ == IVALID_HANDLE) {
    remote_handle_ = IVALID_HANDLE;
    return false;
  } else {
    return true;
  }
}

bool BaseHandle::Close() {
  bool status = true;
  if (Valid()) {
    int ret = func_close_(remote_handle_);
    remote_handle_ = IVALID_HANDLE;
    if (ret != 0) {
      status = false;
    }
  }

  return status;
}

bool BaseHandle::Valid() {
  return (remote_handle_ != IVALID_HANDLE);
}

}  // namespace stub
}  // namespace rpc
