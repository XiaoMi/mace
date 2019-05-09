#!/usr/bin/env sh

set -e

# build for arm linux aarch64
BUILD_DIR=cmake-build/aarch64-linux-gnu-full
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DCROSSTOOL_ROOT=${LINARO_AARCH64_LINUX_GNU} \
      -DCMAKE_TOOLCHAIN_FILE=../../cmake/toolchains/aarch64-linux-gnu.cmake \
      -DCMAKE_BUILD_TYPE=Release          \
      -DMACE_ENABLE_NEON=ON               \
      -DMACE_ENABLE_QUANTIZE=ON           \
      -DMACE_ENABLE_OPENCL=ON             \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DCMAKE_INSTALL_PREFIX=install      \
      ../..
make -j6 VERBOSE=1 && make install
cd ../..
