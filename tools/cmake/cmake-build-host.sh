#! /bin/bash

set -e

# build for host
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/host
fi

MACE_ENABLE_CPU=ON

MACE_ENABLE_CODE_MODE=OFF
if [[ "$RUNMODE" == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

DMACE_ENABLE_BFLOAT16=OFF
if [[ "$BFLOAT16" == "ON" ]]; then
    DMACE_ENABLE_BFLOAT16=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DMACE_ENABLE_NEON=OFF         \
      -DMACE_ENABLE_QUANTIZE=OFF     \
      -DMACE_ENABLE_OPENCL=OFF       \
      -DMACE_ENABLE_BFLOAT16=${DMACE_ENABLE_BFLOAT16}     \
      -DMACE_ENABLE_TESTS=ON         \
      -DMACE_ENABLE_BENCHMARKS=ON    \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}    \
      -DCMAKE_INSTALL_PREFIX=install \
      ../../..
make -j$(nproc) && make install
cd ../../..
