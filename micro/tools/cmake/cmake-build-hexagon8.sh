#!/bin/bash

# HEXAGON_TOOLS is the path of "HEXAGON_Tools/8.*.*" toolchain
if [ -z "$HEXAGON_TOOLS" ]; then
  echo "HEXAGON_TOOLS is undefined";
fi

# HEXAGON_SDK_ROOT is the path of "Hexagon_SDK/3.*.*"
if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "HEXAGON_SDK_ROOT is undefined";
fi

BUILD_DIR=build/micro/hexagon8
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../../micro \
  -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT} \
  -DHEXAGON_TOOLS=${HEXAGON_TOOLS} \
  -DMACE_MICRO_ENABLE_EXAMPLES=OFF \
  -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain/hexagon8.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=install \
  $@ || exit 1

cmake --build . --target install -- -j || exit 1

cd ../../..
