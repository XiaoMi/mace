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

#ifndef MACE_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_
#define MACE_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "mace/core/buffer.h"

namespace mace {

class ScratchImageManager {
 public:
  ScratchImageManager();
  ~ScratchImageManager();

  Image *Spawn(Allocator *allocator,
               const std::vector<size_t> &shape,
               const DataType dt,
               int *id);

  void Deactive(int id);

 private:
  std::unordered_map<int, std::unique_ptr<Image>> images_;
  std::vector<int> reference_count_;
};

class ScratchImage {
 public:
  explicit ScratchImage(ScratchImageManager *);
  ~ScratchImage();

  Image *Scratch(Allocator *allocator,
                 const std::vector<size_t> &shape,
                 const DataType dt);

 private:
  ScratchImageManager *manager_;
  int id_;
};

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_
