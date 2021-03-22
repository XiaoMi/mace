set(RPCMEMS_SRCS
  qualcomm/qualcomm_rpcmem.cc
  rpcmem_factory.cc)

if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(RPCMEMS_SRCS ${RPCMEMS_SRCS} mtk/mtk_rpcmem.cc mtk/mtk_ion_wrapper.cc)
endif(ANDROID_ABI STREQUAL "arm64-v8a")

add_library(rpcmems STATIC ${RPCMEMS_SRCS})
target_link_libraries(rpcmems PRIVATE rpcmem)

#if(ANDROID_ABI STREQUAL "arm64-v8a")
#  target_link_libraries(rpcmems PRIVATE mtk_rpcmem})
#endif(ANDROID_ABI STREQUAL "arm64-v8a")

install(TARGETS rpcmems ARCHIVE DESTINATION lib)
