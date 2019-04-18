// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_PORT_PORT_ARCH_H_
#define MACE_PORT_PORT_ARCH_H_

#if defined __APPLE__
# define MACE_OS_MAC 1
# if TARGET_OS_IPHONE
#  define MACE_OS_IOS 1
# endif
#elif defined __linux__
# define MACE_OS_LINUX 1
# if defined(__ANDROID__) || defined(ANDROID)
#  define MACE_OS_LINUX_ANDROID 1
# endif
#endif

#endif  // MACE_PORT_PORT_ARCH_H_
