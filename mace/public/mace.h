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

enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4, APU = 5 };

enum class DataFormat {
  NONE = 0, NHWC = 1, NCHW = 2,
  HWOI = 100, OIHW = 101, HWIO = 102, OHWI = 103,
  AUTO = 1000,
};

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

// AFFINITY_NONE: initiate 'num_threads_hint' threads with no affinity
// scheduled.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
// AFFINITY_BIG_ONLY: all available big cores are used, and number of threads
// is equal to numbers of available big cores.
// AFFINITY_LITTLE_ONLY: all available little cores are used, and number of
// threads is equal to numbers of available little cores.
// AFFINITY_HIGH_PERFORMANCE: initiate 'num_threads_hint' threads on different
// cores with top-num_threads_hint frequencies.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
// AFFINITY_POWER_SAVE: initiate 'num_threads_hint' threads on different
// cores with bottom-num_threads_hint frequencies.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
enum CPUAffinityPolicy {
  AFFINITY_NONE = 0,
  AFFINITY_BIG_ONLY = 1,
  AFFINITY_LITTLE_ONLY = 2,
  AFFINITY_HIGH_PERFORMANCE = 3,
  AFFINITY_POWER_SAVE = 4,
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

/// Consistent with Android NNAPI
struct PerformanceInfo {
  // Time of executing some workload(millisecond).
  // negative value for unsupported.
  float exec_time;
};

struct Capability {
  // Performance of running with float32 data type
  // run time of the workload for CPU device,
  // ratio of run time to execute same workload compared to the time the CPU
  // execute same workload.
  PerformanceInfo float32_performance;

  // Performance of running with quantized-8 data type
  // ratio compared with float32_performance
  PerformanceInfo quantized8_performance;

  // support or not
  bool supported;
};

/// Get Devices Capacity
///
/// The float32_performance of CPU and GPU is tested using the workload of
/// first 8 layer of mobilenet-v2 which contain Conv(1x1, 3x3),
/// DepthwiseConv(3x3) and ElementWise Ops.
/// The quantized8_performance is just a arbitrary value tested
/// using mobilenet-v2 offline
/// Actually, It's hard to test the precise performance, the result could be
/// more accurate when your model is like with mobilenet-v2, otherwise the
/// value is just a reference.
///
/// \return capability of the device
MACE_API Capability GetCapability(DeviceType device_type,
                                  float cpu_float32_exec_time = 1.f);

MACE_API const char *MaceVersion();

class MaceStatus {
 public:
  enum Code {
    MACE_SUCCESS = 0,
    MACE_INVALID_ARGS = 1,
    MACE_OUT_OF_RESOURCES = 2,
    MACE_UNSUPPORTED = 3,
    MACE_RUNTIME_ERROR = 4,
  };

 public:
  MaceStatus();
  MaceStatus(const Code code);  // NOLINT(runtime/explicit)
  MaceStatus(const Code code, const std::string &information);
  MaceStatus(const MaceStatus &);
  MaceStatus(MaceStatus &&);
  MaceStatus &operator=(const MaceStatus &);
  MaceStatus &operator=(const MaceStatus &&);
  ~MaceStatus();
  Code code() const;
  std::string information() const;

  bool operator==(const MaceStatus &other) const;
  bool operator!=(const MaceStatus &other) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief GPU context contain the status used for GPU device.
///
/// There are some data in common between different MaceEngines using GPU,
/// use one GPUContext could avoid duplication.
///
/// Thread-safe.
/// You could use one GPUContext for multiple parallel MaceEngines.
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
  GPUContextBuilder(GPUContextBuilder &&) = delete;
  GPUContextBuilder &operator=(const GPUContextBuilder &) = delete;
  GPUContextBuilder &operator=(GPUContextBuilder &&) = delete;

  /// \brief Set internal storage factory to store internal data.
  ///
  /// Now the path is used to store the built OpenCL binaries to file,
  /// which could speed up the GPU initialization and first run.
  /// If do not call this API, the initialization maybe slow for GPU.
  ///
  /// \param path  Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetStoragePath(const std::string &path);
  /// \brief Set paths of generated OpenCL compiled kernel binary file (not libOpenCL.so)  // NOLINT(whitespace/line_length)
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the initialization.  // NOLINT(whitespace/line_length)
  /// OpenCL binary is corresponding to the OpenCL Driver version,
  /// you should update the binary when OpenCL Driver changed.
  ///
  /// \param paths MACE will use first file found in all paths
  /// \return
  GPUContextBuilder &SetOpenCLBinaryPaths(
      const std::vector<std::string> &paths);

  /// \brief Set generated OpenCL compiled kernel binary with bytes array
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the initialization.  // NOLINT(whitespace/line_length)
  /// OpenCL binary is corresponding to the OpenCL Driver version,
  /// you should update the binary when OpenCL Driver changed.
  ///
  /// \param data Byte stream of OpenCL binary file
  /// \param size Size of byte stream (data)
  /// \return
  GPUContextBuilder &SetOpenCLBinary(const unsigned char *data,
                                     const size_t size);
  /// \brief Set the path of generated OpenCL parameter file
  ///
  /// If you use GPU for specific soc, the parameters is the local work group
  /// size tuned for specific SOC, which may be faster than the
  /// general parameters.
  ///
  /// \param path Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetOpenCLParameterPath(const std::string &path);
  /// \brief Set generated OpenCL parameter with bytes array
  ///
  /// If you use GPU for specific soc, the parameters is the local work group
  /// size tuned for specific SOC, which may be faster than the
  /// general parameters.
  ///
  /// \param data Byte stream of OpenCL parameter file
  /// \param size Size of byte stream (data)
  /// \return
  GPUContextBuilder &SetOpenCLParameter(const unsigned char *data,
                                        const size_t size);

  std::shared_ptr<GPUContext> Finalize();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class MACE_API MaceEngineConfig {
  friend class MaceEngine;

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
  /// \return MaceStatus::MACE_SUCCESS for success, other for failed.
  MaceStatus SetGPUContext(std::shared_ptr<GPUContext> context);

  /// \brief Set GPU hints, currently only supports Adreno GPU.
  ///
  /// Caution: this function may hurt performance
  /// if improper parameters provided.
  ///
  /// \param perf_hint  performance hint
  /// \param priority_hint  priority hint
  /// \return MaceStatus::MACE_SUCCESS for success, other for failed.
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
  /// \return MaceStatus::MACE_SUCCESS for success, other for failed.
  MaceStatus SetCPUThreadPolicy(int num_threads_hint,
                                CPUAffinityPolicy policy);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// MACE input/output tensor
class MACE_API MaceTensor {
  friend class MaceEngine;

 public:
  // shape - the shape of the tensor, with size n, if shape is unknown
  // in advance, it should be specified large enough to hold tensor of all
  // possible size.
  // data - the buffer of the tensor, must not be null with size equals
  //        shape[0] * shape[1] * ... * shape[n-1].
  //        If you want to pass a buffer which is unsuitable to use the default
  //        shared_ptr deleter (for example, the buffer is not dynamically
  //        allocated by C++, e.g. a C buffer), you can set customized deleter
  //        of shared_ptr and manage the life cycle of the buffer by yourself.
  //        For example, std::shared_ptr<float>(raw_buffer, [](float *){});
  MaceTensor(const std::vector<int64_t> &shape,
             std::shared_ptr<void> data,
             const DataFormat format = DataFormat::NHWC);
  MaceTensor();
  MaceTensor(const MaceTensor &other);
  MaceTensor(const MaceTensor &&other);
  MaceTensor &operator=(const MaceTensor &other);
  MaceTensor &operator=(const MaceTensor &&other);
  ~MaceTensor();

  // shape will be updated to the actual output shape after running.
  const std::vector<int64_t> &shape() const;
  const std::shared_ptr<float> data() const;
  std::shared_ptr<float> data();
  template <typename T>
  const std::shared_ptr<T> data() const {
    return std::static_pointer_cast<T>(raw_data());
  }
  template <typename T>
  std::shared_ptr<T> data() {
    return std::static_pointer_cast<T>(raw_mutable_data());
  }
  DataFormat data_format() const;

 private:
  std::shared_ptr<void> raw_data() const;
  std::shared_ptr<void> raw_mutable_data();

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

/// \brief Create MaceEngine from model graph proto and weights data
///
/// Create MaceEngine object
///
/// \param model_graph_proto[in]: the content of model graph proto
/// \param model_graph_proto_size[in]: the size of model graph proto
/// \param model_weights_data[in]: the content of model weights data, the
///                                returned engine will refer to this buffer
///                                if CPU runtime is used. In this case, the
///                                buffer should keep alive.
/// \param model_weights_data_size[in]: the size of model weights data
/// \param input_nodes[in]: the array of input nodes' name
/// \param output_nodes[in]: the array of output nodes' name
/// \param config[in]: configurations for MaceEngine.
/// \param engine[out]: output MaceEngine object
/// \return MaceStatus::MACE_SUCCESS for success,
///         MaceStatus::MACE_INVALID_ARGS for wrong arguments,
///         MaceStatus::MACE_OUT_OF_RESOURCES for resources is out of range.
MACE_API MaceStatus CreateMaceEngineFromProto(
    const unsigned char *model_graph_proto,
    const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine);

/// \brief Create MaceEngine from files (model file + data file)
/// Deprecated, will be removed in future version
///
/// Create MaceEngine object
///
/// \param model_pb[in]: the content of model graph file
/// \param model_data_file[in]: the path of model data file
/// \param input_nodes[in]: the array of input nodes' name
/// \param output_nodes[in]: the array of output nodes' name
/// \param config[in]: configurations for MaceEngine.
/// \param engine[out]: output MaceEngine object
/// \return MaceStatus::MACE_SUCCESS for success,
///         MaceStatus::MACE_INVALID_ARGS for wrong arguments,
///         MaceStatus::MACE_OUT_OF_RESOURCES for resources is out of range.
MACE_API MaceStatus CreateMaceEngineFromProto(
    const std::vector<unsigned char> &model_pb,
    const std::string &model_data_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine) __attribute__((deprecated));

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_H_
