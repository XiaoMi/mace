#!/bin/bash

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/host
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake ../../..
make mace_proto_py micro_mem_proto_py -j
cd ../../..
