// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

// This file defines core MACE APIs.
// There APIs will be stable and backward compatible.

#ifndef MACE_PUBLIC_MACE_H_
#define MACE_PUBLIC_MACE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef MACE_API
#define MACE_API __attribute__((visibility("default")))
#endif

namespace mace {

class NetDef;

enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3 };

enum GPUPerfHint {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
};

enum GPUPriorityHint {
  PRIORITY_DEFAULT = 0,
  PRIORITY_LOW = 1,
  PRIORITY_NORMAL = 2,
  PRIORITY_HIGH = 3
};

enum CPUAffinityPolicy {
  AFFINITY_NONE = 0,
  AFFINITY_BIG_ONLY = 1,
  AFFINITY_LITTLE_ONLY = 2,
};

struct CallStats {
  int64_t start_micros;
  int64_t end_micros;
};

struct ConvPoolArgs {
  std::vector<int> strides;
  int padding_type;
  std::vector<int> paddings;
  std::vector<int> dilations;
  std::vector<int64_t> kernels;
};

struct OperatorStats {
  std::string operator_name;
  std::string type;
  std::vector<std::vector<int64_t>> output_shape;
  ConvPoolArgs args;
  CallStats stats;
};

class RunMetadata {
 public:
  std::vector<OperatorStats> op_stats;
};

const char *MaceVersion();

enum MaceStatus {
  MACE_SUCCESS = 0,
  MACE_INVALID_ARGS = 1,
  MACE_OUT_OF_RESOURCES = 2
};

#define MACE_RETURN_IF_ERROR(stmt)                                          \
  {                                                                        \
    MaceStatus status = (stmt);                                            \
    if (status != MACE_SUCCESS) {                                          \
      VLOG(0) << "Mace runtime failure: " << __FILE__ << ":" << __LINE__;  \
      return status;                                                       \
    }                                                                      \
  }

/// \brief Get ARM big.LITTLE configuration.
///
/// This function will detect the max frequencies of all CPU cores, and assume
/// the cores with largest max frequencies as big cores, and all the remaining
/// cores as little. If all cpu core's max frequencies equals, big_core_ids and
/// little_core_ids will both be filled with all cpu core ids.
///
/// \param [out] big_core_ids
/// \param [out] little_core_ids
/// \return If successful, it returns MACE_SUCCESS and error if it can't
///         reliabley detect the frequency of big-LITTLE cores (e.g. MTK).

MACE_API MaceStatus GetBigLittleCoreIDs(std::vector<int> *big_core_ids,
                                        std::vector<int> *little_core_ids);

/// \brief GPU context contain the status used for GPU device.
///
/// The life cycle of GPUContext object is the same as MaceEngines use it.
/// Just use one GPUContext for all MaceEngines, which will speed up the
/// initialization procedure. There are some data in common between different
/// MaceEngines using GPU, use one GPUContext could avoid duplication.
class GPUContext;

/// \brief GPUContext builder.
///
/// Use the GPUContextBuilder to generate GPUContext.
/// Not thread-safe
class MACE_API GPUContextBuilder {
 public:
  GPUContextBuilder();
  ~GPUContextBuilder();
  GPUContextBuilder(const GPUContextBuilder &) = delete;
  GPUContextBuilder(const GPUContextBuilder &&) = delete;
  GPUContextBuilder &operator=(const GPUContextBuilder &) = delete;
  GPUContextBuilder &operator=(const GPUContextBuilder &&) = delete;

  /// \brief Set internal storage factory to store internal data.
  ///
  /// Now the path is used to store the built OpenCL binaries to file,
  /// which could speed up the GPU initialization and first run.
  /// If do not call this API, the initialization maybe slow for GPU.
  ///
  /// \param path  Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetStoragePath(const std::string &path);
  /// \brief Set paths of Generated OpenCL Compiled Kernel Binary file (not libOpenCL.so)  // NOLINT(whitespace/line_length)
  ///
  /// if you use gpu of specific soc, Using OpenCL binary will speed up the initialization.  // NOLINT(whitespace/line_length)
  /// OpenCL binary is corresponding to the OpenCL Driver version,
  /// you should update the binary when OpenCL Driver changed.
  ///
  /// \param paths MACE will use first file found in all paths
  /// \return
  GPUContextBuilder &SetOpenCLBinaryPaths(
      const std::vector<std::string> &paths);
  /// \brief Set the path of Generated OpenCL parameter file
  ///
  /// If you use gpu for specific soc, The parameters is the local work group
  /// size tuned for specific SOC, which may be faster than the
  /// general parameters.
  ///
  /// \param path Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetOpenCLParameterPath(const std::string &path);

  std::shared_ptr<GPUContext> Finalize();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class MACE_API MaceEngineConfig {
 public:
  explicit MaceEngineConfig(const DeviceType device_type);
  ~MaceEngineConfig();
  MaceEngineConfig(const MaceEngineConfig &) = delete;
  MaceEngineConfig(const MaceEngineConfig &&) = delete;
  MaceEngineConfig &operator=(const MaceEngineConfig &) = delete;
  MaceEngineConfig &operator=(const MaceEngineConfig &&) = delete;

  /// \brief Set GPUContext
  ///
  /// Just use one GPUContext for multiple models run on GPU.
  /// \param context created use GPUContextBuilder
  /// \return MACE_SUCCESS for success, other for failed.
  MaceStatus SetGPUContext(std::shared_ptr<GPUContext> context);

  /// \brief Set GPU hints, currently only supports Adreno GPU.
  ///
  /// Caution: this function may hurt performance
  /// if improper parameters provided.
  ///
  /// \param perf_hint  performance hint
  /// \param priority_hint  priority hint
  /// \return MACE_SUCCESS for success, other for failed.
  MaceStatus SetGPUHints(GPUPerfHint perf_hint,
                         GPUPriorityHint priority_hint);

  /// \brief Set CPU threads number and affinity policy.
  ///
  /// Caution: this function may hurt performance if improper
  /// parameters provided. When num_threads_hint is zero or negative,
  /// the function will set the threads number equaling to the number of
  /// big (AFFINITY_BIG_ONLY), little (AFFINITY_LITTLE_ONLY) or all
  /// (AFFINITY_NONE) cores according to the policy. The threads number will
  /// also be truncated to the corresponding cores number when num_threads_hint
  /// is larger than it.
  /// The OpenMP threads will be bind to (via sched_setaffinity) big cores
  /// (AFFINITY_BIG_ONLY) and little cores (AFFINITY_LITTLE_ONLY).
  ///
  /// \param num_threads_hint it is only a hint.
  /// \param policy one of CPUAffinityPolicy
  /// \param status MACE_SUCCESS for successful, or it can't reliabley
  /// detect big-LITTLE cores (see GetBigLittleCoreIDs). In such cases, it's
  /// suggested to use AFFINITY_NONE to use all cores.
  /// \param use_gemmlowp use gemmlowp for quantized inference
  /// \return MACE_SUCCESS for success, other for failed.
  MaceStatus SetCPUThreadPolicy(int num_threads_hint,
                                CPUAffinityPolicy policy,
                                bool use_gemmlowp = false);

  /// \brief Set OpenMP threads number and processor affinity.
  ///
  /// Caution: this function may hurt performance
  /// if improper parameters provided.
  /// This function may not work well on some chips (e.g. MTK). Setting thread
  /// affinity to offline cores may run very slow or unexpectedly.
  /// In such cases, please use SetOpenMPThreadPolicy with default policy
  /// instead.
  ///
  /// \param num_threads
  /// \param cpu_ids
  /// \return MACE_SUCCESS for success, other for failed.
  MaceStatus SetOpenMPThreadAffinity(
      int num_threads,
      const std::vector<int> &cpu_ids);

  DeviceType device_type() const;

  int num_threads() const;

  std::shared_ptr<GPUContext> gpu_context() const;

  GPUPriorityHint gpu_priority_hint() const;

  GPUPerfHint gpu_perf_hint() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// MACE input/output tensor
class MACE_API MaceTensor {
 public:
  // shape - the shape of the tensor, with size n
  // data - the buffer of the tensor, must not be null with size equals
  //        shape[0] * shape[1] * ... * shape[n-1]
  MaceTensor(const std::vector<int64_t> &shape,
             std::shared_ptr<float> data);
  MaceTensor();
  MaceTensor(const MaceTensor &other);
  MaceTensor(const MaceTensor &&other);
  MaceTensor &operator=(const MaceTensor &other);
  MaceTensor &operator=(const MaceTensor &&other);
  ~MaceTensor();

  const std::vector<int64_t> &shape() const;
  const std::shared_ptr<float> data() const;
  std::shared_ptr<float> data();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class MACE_API MaceEngine {
 public:
  explicit MaceEngine(const MaceEngineConfig &config);
  ~MaceEngine();

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data);

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const std::string &model_data_file);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  MaceEngine(const MaceEngine &) = delete;
  MaceEngine &operator=(const MaceEngine &) = delete;
};

/// \brief Create MaceEngine from files (model file + data file)
///
/// Create MaceEngine object
///
/// \param model_pb[in]: the content of model graph file
/// \param model_data_file[in]: the path of model data file
/// \param input_nodes[in]: the array of input nodes' name
/// \param output_nodes[in]: the array of output nodes' name
/// \param config[in]: configurations for MaceEngine.
/// \param engine[out]: output MaceEngine object
/// \return MACE_SUCCESS for success, MACE_INVALID_ARGS for wrong arguments,
///         MACE_OUT_OF_RESOURCES for resources is out of range.
MACE_API MaceStatus CreateMaceEngineFromProto(
    const std::vector<unsigned char> &model_pb,
    const std::string &model_data_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine);

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_H_
