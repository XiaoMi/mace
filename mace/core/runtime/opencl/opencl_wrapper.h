//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_WRAPPER_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_WRAPPER_H_

namespace mace {

// These functions are not thread-safe.
void LoadOpenCLLibrary();
void UnloadOpenCLLibrary();

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_WRAPPER_H_
