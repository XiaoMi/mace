#!/usr/bin/env sh

set -e

# build for arm linux gnueabihf
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/arm-linux-gnueabihf
fi

MACE_ENABLE_CODE_MODE=OFF
if [[ $RUNMODE == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

MACE_ENABLE_OPENCL=OFF
if [[ "$RUNTIME" == "GPU" ]]; then
    MACE_ENABLE_OPENCL=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DCROSSTOOL_ROOT=${LINARO_ARM_LINUX_GNUEABIHF} \
      -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchains/arm-linux-gnueabihf.cmake \
      -DCMAKE_BUILD_TYPE=Release          \
      -DMACE_ENABLE_NEON=ON               \
      -DMACE_ENABLE_QUANTIZE=ON           \
      -DMACE_ENABLE_OPENCL=${MACE_ENABLE_OPENCL}              \
      -DMACE_ENABLE_BFLOAT16=ON           \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DMACE_ENABLE_TESTS=ON              \
      -DMACE_ENABLE_BENCHMARKS=ON         \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}        \
      -DCMAKE_INSTALL_PREFIX=install      \
      ../../..
make -j6 VERBOSE=1 && make install
cd ../../..
