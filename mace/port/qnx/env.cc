#include <errno.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <sys/neutrino.h>
#include <sys/syspage.h>

#include "mace/port/qnx/env.h"
#include "mace/port/env.h"
#include "mace/port/posix/file_system.h"
#include "mace/port/posix/time.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {

int64_t QnxEnv::NowMicros() {
	return mace::port::posix::NowMicros();
}

MaceStatus QnxEnv::AdviseFree(void *addr, size_t length) {
	MACE_UNUSED(addr);
	MACE_UNUSED(length);
  return MaceStatus::MACE_SUCCESS;
}

FileSystem *QnxEnv::GetFileSystem() {
	return &posix_file_system_;
}

MaceStatus QnxEnv::GetCPUMaxFreq(std::vector<float> *max_freqs) {
	// small cores
	max_freqs->push_back(1785608000);
	max_freqs->push_back(1785608000);
	max_freqs->push_back(1785608000);
	max_freqs->push_back(1785608000);
	// big cores
	max_freqs->push_back(2131184000);
	max_freqs->push_back(2131184000);
	max_freqs->push_back(2131184000);
	// super big core
	max_freqs->push_back(2419184000);
	return MaceStatus::MACE_SUCCESS;
}

MaceStatus QnxEnv::SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
	unsigned    num_elements = 0;
	int         *rsizep, masksize_bytes, size;
	unsigned    *rmaskp, *imaskp;
	void        *my_data;

	/* Determine the number of array elements required to hold
	* the runmasks, based on the number of CPUs in the system. */
	num_elements = RMSK_SIZE(_syspage_ptr->num_cpu);

	/* Determine the size of the runmask, in bytes. */
	masksize_bytes = num_elements * sizeof(unsigned);

	/* Allocate memory for the data structure that we'll pass
	* to ThreadCtl(). We need space for an integer (the number
	* of elements in each mask array) and the two masks
	* (runmask and inherit mask). */

	size = sizeof(int) + 2 * masksize_bytes;
	if ((my_data = malloc(size)) == NULL) {
		LOG(WARNING) << "SchedSetAffinity failed: " << strerror(errno);
    return MaceStatus(MaceStatus::MACE_OUT_OF_RESOURCES,
                      "SchedSetAffinity failed: " + std::string(strerror(errno)));
	} 
	memset(my_data, 0x00, size);
	/* Set up pointers to the "members" of the structure. */
	rsizep = (int *)my_data;
	rmaskp = (unsigned *)rsizep + 1;
	imaskp = rmaskp + num_elements;

	/* Set the size. */
	*rsizep = num_elements;

	for (size_t cpu_id : cpu_ids) {
		/* Set the runmask. Call this macro once for each processor
				the thread can run on. */
		RMSK_SET(cpu_id, rmaskp);
		/* Set the inherit mask. Call this macro once for each
				processor the thread's children can run on. */
		RMSK_SET(cpu_id, imaskp);
	}

	if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT, my_data) == -1) {
		LOG(WARNING) << "ThreadCtl failed: " << strerror(errno);
    return MaceStatus(MaceStatus::MACE_RUNTIME_ERROR,
                      "ThreadCtl failed: " + std::string(strerror(errno)));
	}
  return MaceStatus::MACE_SUCCESS;
}

LogWriter *QnxEnv::GetLogWriter() {
	return &log_writer_;
}

std::vector<std::string> QnxEnv::GetBackTraceUnsafe(int max_steps) {
	return std::vector<std::string>();
}

std::unique_ptr<MallocLogger> QnxEnv::NewMallocLogger(
		std::ostringstream *oss,
		const std::string &name) {
	return make_unique<MallocLogger>();
}

Env *Env::Default() {
  static QnxEnv qnx_env;
  return &qnx_env;
}

}  // namespace port
}  // namespace mace