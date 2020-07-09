set(CMAKE_C_COMPILER "${HEXAGON_TOOLS}/qc/bin/hexagon-clang")
set(CMAKE_CXX_COMPILER "${HEXAGON_TOOLS}/qc/bin/hexagon-clang")
set(CMAKE_AR "${HEXAGON_TOOLS}/gnu/bin/hexagon-ar" CACHE FILEPATH "Archiver")
set(CMAKE_LINKER "${HEXAGON_TOOLS}/qc/bin/hexagon-llvm-link")

set(CMAKE_FIND_ROOT_PATH "${HEXAGON_TOOLS}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(HEXAGON ON)
set(HEXAGON6 ON)

if (NOT TARGET hexagon_sdk_headers_dsp)
  add_library(hexagon_sdk_headers_dsp INTERFACE)
  target_include_directories(hexagon_sdk_headers_dsp INTERFACE ${HEXAGON_SDK_ROOT}/incs)
  target_include_directories(hexagon_sdk_headers_dsp INTERFACE ${HEXAGON_SDK_ROOT}/incs/stddef)
  target_include_directories(hexagon_sdk_headers_dsp INTERFACE ${HEXAGON_SDK_ROOT}/libs/common/remote/ship/hexagon_Release_toolv81_v60)
endif()

if (NOT TARGET hexagon_sdk_headers_arm)
  add_library(hexagon_sdk_headers_arm INTERFACE)
  target_include_directories(hexagon_sdk_headers_arm INTERFACE ${HEXAGON_SDK_ROOT}/incs)
  target_include_directories(hexagon_sdk_headers_arm INTERFACE ${HEXAGON_SDK_ROOT}/incs/stddef)
  target_include_directories(hexagon_sdk_headers_arm INTERFACE ${HEXAGON_SDK_ROOT}/libs/common/remote/ship/hexagon_Release_toolv81_v60)
endif()

if (NOT TARGET hexagon_sdk_cdsprpc)
  add_library(hexagon_sdk_cdsprpc SHARED IMPORTED)
  set_target_properties(hexagon_sdk_cdsprpc
    PROPERTIES IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_Release_aarch64/libcdsprpc.so"
  )
endif()

if (NOT TARGET hexagon_sdk_rpcmem)
  add_library(hexagon_sdk_rpcmem STATIC IMPORTED)
  set_target_properties(hexagon_sdk_rpcmem
    PROPERTIES IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/libs/common/rpcmem/rpcmem.a"
  )
endif()
