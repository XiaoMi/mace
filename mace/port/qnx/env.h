#ifndef MACE_PORT_QNX_ENV_H_
#define MACE_PORT_QNX_ENV_H_

#include <vector>

#include "mace/port/env.h"
#include "mace/port/posix/file_system.h"
#include "mace/port/logger.h"

namespace mace {
namespace port {

class QnxEnv : public Env {
public:
  int64_t NowMicros() override;
  MaceStatus AdviseFree(void *addr, size_t length) override;
  MaceStatus GetCPUMaxFreq(std::vector<float> *max_freqs) override;
  MaceStatus SchedSetAffinity(const std::vector<size_t> &cpu_ids) override;
  FileSystem *GetFileSystem() override;
	LogWriter *GetLogWriter() override;
  std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;
  std::unique_ptr<MallocLogger> NewMallocLogger(
      std::ostringstream *oss,
      const std::string &name) override;

protected:
  PosixFileSystem posix_file_system_;
private:
	LogWriter log_writer_;
};

}  // namespace port
}  // namespace mace

#endif  // MACE_PORT_QNX_ENV_H_