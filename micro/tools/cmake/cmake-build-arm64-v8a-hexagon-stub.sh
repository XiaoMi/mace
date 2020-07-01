#!/bin/bash
if [ -z "$ANDROID_NDK_HOME" ]; then
  echo "ANDROID_NDK_HOME is undefined";
fi

if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "HEXAGON_SDK_ROOT is undefined";
fi

BUILD_DIR=build/cmake-build/arm64-v8a
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../.. \
  -DANDROID_ABI="arm64-v8a" \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
  -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT} \
  -DANDROID_NATIVE_API_LEVEL=21       \
  -DCMAKE_BUILD_TYPE=Release          \
  -DANDROID_STL=c++_shared            \
  -DMACE_ENABLE_RPCMEM=ON             \
  -DCMAKE_INSTALL_PREFIX=install      \
  -DHEXAGON_STUB=ON \
  $@ || exit 1

cmake --build . -- -j || exit 1

cd ../../..
