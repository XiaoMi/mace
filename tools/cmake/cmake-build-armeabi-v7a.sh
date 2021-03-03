#! /bin/bash

set -e

# build for android armeabi-v7a
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/armeabi-v7a
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

MACE_ENABLE_QUANTIZE=OFF
if [[ "$QUANTIZE" == "ON" ]]; then
    MACE_ENABLE_QUANTIZE=ON
fi

DMACE_ENABLE_BFLOAT16=OFF
if [[ "$BFLOAT16" == "ON" ]]; then
    DMACE_ENABLE_BFLOAT16=ON
fi

MACE_ENABLE_RPCMEM=OFF
if [[ "$RPCMEM" == "ON" ]]; then
    MACE_ENABLE_RPCMEM=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DANDROID_ABI="armeabi-v7a" \
      -DANDROID_ARM_NEON=ON \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
      -DANDROID_NATIVE_API_LEVEL=21                          \
      -DCMAKE_BUILD_TYPE=Release                             \
      -DANDROID_STL=c++_shared                               \
      -DMACE_ENABLE_NEON=ON                                  \
      -DMACE_ENABLE_QUANTIZE=${MACE_ENABLE_QUANTIZE}         \
      -DMACE_ENABLE_OPENCL=${MACE_ENABLE_OPENCL}             \
      -DMACE_ENABLE_HEXAGON_DSP=${MACE_ENABLE_HEXAGON_DSP}   \
      -DMACE_ENABLE_HEXAGON_HTA=${MACE_ENABLE_HEXAGON_HTA}   \
      -DMACE_ENABLE_MTK_APU=${MACE_ENABLE_MTK_APU}           \
      -DMACE_ENABLE_BFLOAT16=${DMACE_ENABLE_BFLOAT16}        \
      -DMACE_ENABLE_OPT_SIZE=ON                              \
      -DMACE_ENABLE_OBFUSCATE=ON                             \
      -DMACE_ENABLE_TESTS=ON                                 \
      -DMACE_ENABLE_BENCHMARKS=ON                            \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}       \
      -DMACE_ENABLE_RPCMEM=${MACE_ENABLE_RPCMEM}             \
      -DCMAKE_INSTALL_PREFIX=install                         \
      ../../..
make -j$(nproc)1 && make install
cd ../../..
