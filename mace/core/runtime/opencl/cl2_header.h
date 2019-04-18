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

#ifndef MACE_CORE_RUNTIME_OPENCL_CL2_HEADER_H_
#define MACE_CORE_RUNTIME_OPENCL_CL2_HEADER_H_

// Do not include cl2.hpp directly, include this header instead.

#include "mace/port/port-arch.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#ifdef MACE_OS_MAC
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 200
#endif  // MACE_OS_MAC

#ifdef MACE_OS_MAC
// disable deprecated warning in macOS 10.14
#define CL_SILENCE_DEPRECATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif  // MACE_OS_MAC

#include "CL/cl2.hpp"

#ifdef MACE_OS_MAC
#pragma GCC diagnostic pop
#endif

#endif  // MACE_CORE_RUNTIME_OPENCL_CL2_HEADER_H_
