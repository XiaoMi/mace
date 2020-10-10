#!/bin/bash

if [ -z "$ANDROID_NDK_HOME" ]; then
  echo "ANDROID_NDK_HOME is undefined";
  exit -1;
fi

if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "HEXAGON_SDK_ROOT is undefined";
  exit -1;
fi

BUILD_DIR=build/micro/arm64-v8a
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../../micro \
  -DANDROID_ABI="arm64-v8a" \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
  -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT} \
  -DANDROID_NATIVE_API_LEVEL=21       \
  -DCMAKE_BUILD_TYPE=Release          \
  -DANDROID_STL=c++_shared            \
  -DMACE_ENABLE_RPCMEM=ON             \
  -DCMAKE_INSTALL_PREFIX=install      \
  -DMACE_MICRO_ENABLE_EXAMPLES=OFF \
  -DHEXAGON_STUB=ON \
  $@ || exit 1

cmake --build . --target install --target install -- -j || exit 1

cd ../../..
