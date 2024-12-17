add_library(utils STATIC
  string_util.cc
  thread_pool.cc
  status.cc
  statistics.cc
)

if(NOT ANDROID AND NOT WIN32 AND NOT QNX)
  target_link_libraries(utils PUBLIC pthread)
endif(NOT ANDROID AND NOT WIN32 AND NOT QNX)
