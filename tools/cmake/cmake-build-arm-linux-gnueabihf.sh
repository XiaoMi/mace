#! /bin/bash

set -e

# build for arm linux gnueabihf
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/arm-linux-gnueabihf
fi

MACE_ENABLE_CODE_MODE=OFF
if [[ $RUNMODE == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

MACE_ENABLE_NEON=OFF
MACE_ENABLE_CPU=OFF
if [[ "$RUNTIME" == "CPU" ]]; then
    MACE_ENABLE_NEON=ON
    MACE_ENABLE_CPU=ON
fi

MACE_ENABLE_OPENCL=OFF
if [[ "$RUNTIME" == "GPU" ]]; then
    MACE_ENABLE_OPENCL=ON
fi

MACE_ENABLE_QUANTIZE=OFF
if [[ "$QUANTIZE" == "ON" ]]; then
    MACE_ENABLE_QUANTIZE=ON
fi

DMACE_ENABLE_BFLOAT16=OFF
if [[ "$BFLOAT16" == "ON" ]]; then
    DMACE_ENABLE_BFLOAT16=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DCROSSTOOL_ROOT=${LINARO_ARM_LINUX_GNUEABIHF} \
      -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchains/arm-linux-gnueabihf.cmake \
      -DCMAKE_BUILD_TYPE=Release          \
      -DMACE_ENABLE_NEON=${MACE_ENABLE_NEON}                 \
      -DMACE_ENABLE_QUANTIZE=${MACE_ENABLE_QUANTIZE}         \
      -DMACE_ENABLE_CPU=${MACE_ENABLE_CPU}                   \
      -DMACE_ENABLE_OPENCL=${MACE_ENABLE_OPENCL}             \
      -DMACE_ENABLE_BFLOAT16=${DMACE_ENABLE_BFLOAT16}        \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DMACE_ENABLE_TESTS=ON              \
      -DMACE_ENABLE_BENCHMARKS=ON         \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}       \
      -DCMAKE_INSTALL_PREFIX=install      \
      ../../..
make -j$(nproc) && make install
cd ../../..
