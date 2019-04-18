# Tested with gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabi
# https://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER "${CROSSTOOL_ROOT}/bin/arm-linux-gnueabi-gcc")
set(CMAKE_CXX_COMPILER "${CROSSTOOL_ROOT}/bin/arm-linux-gnueabi-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
