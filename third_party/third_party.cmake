set(MACE_THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party" CACHE STRING "Third party libraries download & build directories.")

# Forwarding the cross compile flags
set(THIRD_PARTY_EXTRA_CMAKE_ARGS
  -DCMAKE_C_FLAGS=${MACE_CC_FLAGS}
  -DCMAKE_CXX_FLAGS=${MACE_CC_FLAGS}
)

if(CMAKE_TOOLCHAIN_FILE)
  set(THIRD_PARTY_EXTRA_CMAKE_ARGS
      ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
  )
endif(CMAKE_TOOLCHAIN_FILE)

if(CROSSTOOL_ROOT)
  set(THIRD_PARTY_EXTRA_CMAKE_ARGS
      ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
      -DCROSSTOOL_ROOT=${CROSSTOOL_ROOT}
  )
endif(CROSSTOOL_ROOT)

if(ANDROID_ABI)
  set(THIRD_PARTY_EXTRA_CMAKE_ARGS
      ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
      -DANDROID_ABI=${ANDROID_ABI}
  )
endif(ANDROID_ABI)

if(ANDROID_NATIVE_API_LEVEL)
  set(THIRD_PARTY_EXTRA_CMAKE_ARGS
      ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
      -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
  )
endif(ANDROID_NATIVE_API_LEVEL)

if(PLATFORM)
  set(THIRD_PARTY_EXTRA_CMAKE_ARGS
      ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
      -DPLATFORM=${PLATFORM}
  )
endif(PLATFORM)

include(${PROJECT_SOURCE_DIR}/third_party/eigen3/eigen3.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/gemmlowp/gemmlowp.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/gflags/gflags.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/half/half.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/opencl-clhpp/opencl-clhpp.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/opencl-headers/opencl-headers.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/protobuf/protobuf.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/tflite/tflite.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/caffe/caffe.cmake)

if(MACE_ENABLE_RPCMEM)
  include(${PROJECT_SOURCE_DIR}/third_party/rpcmem/rpcmem.cmake)
endif(MACE_ENABLE_RPCMEM)

if(MACE_ENABLE_HEXAGON_DSP)
  include(${PROJECT_SOURCE_DIR}/third_party/nnlib/nnlib.cmake)
endif(MACE_ENABLE_HEXAGON_DSP)

if(MACE_ENABLE_HEXAGON_HTA)
  include(${PROJECT_SOURCE_DIR}/third_party/hta/hta.cmake)
endif(MACE_ENABLE_HEXAGON_HTA)

if(MACE_ENABLE_MTK_APU)
  include(${PROJECT_SOURCE_DIR}/third_party/apu/apu.cmake)
endif(MACE_ENABLE_MTK_APU)
