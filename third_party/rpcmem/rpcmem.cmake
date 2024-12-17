set(RPCMEM_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/rpcmem")
set(RPCMEM_INCLUDE_DIR  "${RPCMEM_INSTALL_DIR}")

include_directories(SYSTEM "${RPCMEM_INCLUDE_DIR}")

if(QNX)
set(RPCMEM_LIB "${RPCMEM_INSTALL_DIR}/qnx/libfastrpc_pmem.so")
else()
set(RPCMEM_LIB "${RPCMEM_INSTALL_DIR}/${ANDROID_ABI}/rpcmem.a")
endif()

add_library(rpcmem STATIC IMPORTED GLOBAL)
set_target_properties(rpcmem PROPERTIES IMPORTED_LOCATION ${RPCMEM_LIB})
