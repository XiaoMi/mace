set(RPCMEMS_SRCS
  qualcomm/qualcomm_rpcmem.cc
  dmabufheap/dma_buf_heap_rpcmem.cc
  dmabufheap/dma_buf_heap_wrapper.cc
  rpcmem_factory.cc)

if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(RPCMEMS_SRCS ${RPCMEMS_SRCS} mtk/mtk_rpcmem.cc mtk/mtk_ion_wrapper.cc)
endif()

add_library(rpcmems STATIC ${RPCMEMS_SRCS})
target_link_libraries(rpcmems PRIVATE rpcmem)

install(TARGETS rpcmems ARCHIVE DESTINATION lib)
