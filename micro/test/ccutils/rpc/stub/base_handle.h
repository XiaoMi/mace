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



#ifndef MICRO_TEST_CCUTILS_RPC_STUB_BASE_HANDLE_H_
#define MICRO_TEST_CCUTILS_RPC_STUB_BASE_HANDLE_H_

#include <memory>

#include "remote.h"  // NOLINT

namespace rpc {
namespace stub {

class BaseHandle {
 protected:
  typedef int FuncOpen(const char *name, remote_handle64 *h);
  typedef int FuncClose(remote_handle64 h);
  FuncOpen *func_open_;
  FuncClose *func_close_;
  const char *uri_;
  remote_handle64 remote_handle_;

 public:
  explicit BaseHandle(FuncOpen *func_open,
                      FuncClose *func_close,
                      const char *uri);

  ~BaseHandle();

  bool Open();

  bool Close();

  bool Valid();
};

}  // namespace stub
}  // namespace rpc

#endif  // MICRO_TEST_CCUTILS_RPC_STUB_BASE_HANDLE_H_
