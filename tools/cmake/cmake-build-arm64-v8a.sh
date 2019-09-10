#!/usr/bin/env sh

set -e

# build for android arm64-v8a
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/arm64-v8a
fi

MACE_ENABLE_OPENCL=OFF
MACE_ENABLE_HEXAGON_DSP=OFF
MACE_ENABLE_HEXAGON_HTA=OFF
MACE_ENABLE_MTK_APU=OFF
if [[ "$RUNTIME" == "GPU" ]]; then
    MACE_ENABLE_OPENCL=ON
elif [[ "$RUNTIME" == "HEXAGON" ]]; then
    MACE_ENABLE_HEXAGON_DSP=ON
elif [[ "$RUNTIME" == "HTA" ]]; then
    MACE_ENABLE_HEXAGON_HTA=ON
elif [[ "$RUNTIME" == "APU" ]]; then
    MACE_ENABLE_MTK_APU=ON
fi

MACE_ENABLE_CODE_MODE=OFF
if [[ "$RUNMODE" == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DANDROID_ABI="arm64-v8a" \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
      -DANDROID_NATIVE_API_LEVEL=28       \
      -DCMAKE_BUILD_TYPE=Release          \
      -DCMAKE_ANDROID_STL_TYPE=c++_shared \
      -DMACE_ENABLE_NEON=ON               \
      -DMACE_ENABLE_QUANTIZE=ON           \
      -DMACE_ENABLE_OPENCL=${MACE_ENABLE_OPENCL}             \
      -DMACE_ENABLE_HEXAGON_DSP=${MACE_ENABLE_HEXAGON_DSP}   \
      -DMACE_ENABLE_HEXAGON_HTA=${MACE_ENABLE_HEXAGON_HTA}   \
      -DMACE_ENABLE_MTK_APU=${MACE_ENABLE_MTK_APU}           \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DMACE_ENABLE_TESTS=ON              \
      -DMACE_ENABLE_BENCHMARKS=ON         \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}        \
      -DCMAKE_INSTALL_PREFIX=install      \
      ../../..
make -j6 VERBOSE=1 && make install
cd ../../..
