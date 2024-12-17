#! /bin/bash

set -e

# build for android arm64-v8a
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/arm64-v8a
fi
LIB_DIR=$BUILD_DIR"/install/lib/"
rm -rf $LIB_DIR

MACE_ENABLE_NEON=OFF
MACE_ENABLE_CPU=ON
MACE_ENABLE_OPENCL=OFF
MACE_ENABLE_HEXAGON_DSP=OFF
MACE_ENABLE_HEXAGON_HTA=OFF
MACE_ENABLE_MTK_APU=OFF
MACE_MTK_APU_VERSION=-1
if [[ "$RUNTIME" == "CPU" ]]; then
    MACE_ENABLE_NEON=ON
    MACE_ENABLE_CPU=ON
elif [[ "$RUNTIME" == "GPU" ]]; then
    MACE_ENABLE_NEON=ON
    MACE_ENABLE_CPU=ON
    MACE_ENABLE_OPENCL=ON
elif [[ "$RUNTIME" == "HEXAGON" ]]; then
    MACE_ENABLE_HEXAGON_DSP=ON
elif [[ "$RUNTIME" == "HTA" ]]; then
    MACE_ENABLE_HEXAGON_HTA=ON
    MACE_ENABLE_OPENCL=ON
elif [[ "$RUNTIME" == "HTP" ]]; then
    MACE_ENABLE_QNN=ON
    MACE_ENABLE_OPENCL=ON
elif [[ "$RUNTIME" == "APU" ]]; then
    MACE_ENABLE_MTK_APU=ON
    set +e
    python tools/python/apu_utils.py get-version --target_abi=arm64-v8a
    MACE_MTK_APU_VERSION=`echo $?`
    echo "The apu version is ${MACE_MTK_APU_VERSION}"
    set -e
fi

MACE_ENABLE_CODE_MODE=OFF
if [[ "$RUNMODE" == "code" ]]; then
    MACE_ENABLE_CODE_MODE=ON
fi

MACE_ENABLE_QUANTIZE=OFF
if [[ "$QUANTIZE" == "ON" ]]; then
    MACE_ENABLE_QUANTIZE=ON
fi

MACE_ENABLE_BFLOAT16=OFF
if [[ "$BFLOAT16" == "ON" ]]; then
    MACE_ENABLE_BFLOAT16=ON
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DANDROID_ABI="arm64-v8a" \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
      -DANDROID_NATIVE_API_LEVEL=21       \
      -DCMAKE_BUILD_TYPE=Release          \
      -DANDROID_STL=c++_shared            \
      -DMACE_ENABLE_NEON=${MACE_ENABLE_NEON}                 \
      -DMACE_ENABLE_QUANTIZE=${MACE_ENABLE_QUANTIZE}         \
      -DMACE_ENABLE_CPU=${MACE_ENABLE_CPU}                   \
      -DMACE_ENABLE_OPENCL=${MACE_ENABLE_OPENCL}             \
      -DMACE_ENABLE_HEXAGON_DSP=${MACE_ENABLE_HEXAGON_DSP}   \
      -DMACE_ENABLE_HEXAGON_HTA=${MACE_ENABLE_HEXAGON_HTA}   \
      -DMACE_ENABLE_QNN=${MACE_ENABLE_QNN}                   \
      -DMACE_ENABLE_MTK_APU=${MACE_ENABLE_MTK_APU}           \
      -DMACE_MTK_APU_VERSION=${MACE_MTK_APU_VERSION}         \
      -DMACE_ENABLE_BFLOAT16=${MACE_ENABLE_BFLOAT16}         \
      -DMACE_ENABLE_OPT_SIZE=ON           \
      -DMACE_ENABLE_OBFUSCATE=ON          \
      -DMACE_ENABLE_TESTS=ON              \
      -DMACE_ENABLE_BENCHMARKS=ON         \
      -DMACE_ENABLE_CODE_MODE=${MACE_ENABLE_CODE_MODE}        \
      -DMACE_ENABLE_RPCMEM=ON                                 \
      -DCMAKE_INSTALL_PREFIX=install \
      ../../..
make -j$(nproc) VERBOSE=1 && make install
cd ../../..

# Detect the plugin-device and copy the valid so to the output dir
if [[ "$RUNTIME" == "APU" ]]; then
    python tools/python/apu_utils.py copy-so-files --target_abi arm64-v8a --apu_path $LIB_DIR
fi
