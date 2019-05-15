#!/usr/bin/env sh

set -e

# build for android arm64-v8a
BUILD_DIR=cmake-build/android-arm64-v8a-cpu
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DANDROID_ABI="arm64-v8a" \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
      -DANDROID_NATIVE_API_LEVEL=14       \
      -DCMAKE_BUILD_TYPE=Release          \
      -DCMAKE_ANDROID_STL_TYPE=c++_shared \
      -DMACE_ENABLE_NEON=ON               \
      -DMACE_ENABLE_QUANTIZE=OFF          \
      -DMACE_ENABLE_OPENCL=OFF            \
      -DMACE_ENABLE_HEXAGON_DSP=OFF       \
      -DMACE_ENABLE_HEXAGON_HTA=OFF       \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DCMAKE_INSTALL_PREFIX=install      \
      ../..
make -j6 VERBOSE=1 && make install
cd ../..
