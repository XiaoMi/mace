#!/usr/bin/env bash

set -e -u -o pipefail

pushd ../../../

TARGET_ABI=arm64-v8a
LIBRARY_DIR=mace/examples/android/macelibrary/src/main/cpp/
INCLUDE_DIR=$LIBRARY_DIR/include/mace/public/
LIBMACE_DIR=$LIBRARY_DIR/lib/$TARGET_ABI/

rm -rf $LIBRARY_DIR/include/
mkdir -p $INCLUDE_DIR

rm -rf $LIBRARY_DIR/lib/
mkdir -p $LIBMACE_DIR

rm -rf $LIBRARY_DIR/model/

python tools/converter.py convert --config=mace/examples/android/mobilenet.yml --target_abis=$TARGET_ABI
cp -rf builds/mobilenet/include/mace/public/*.h $INCLUDE_DIR
cp -rf builds/mobilenet/model $LIBRARY_DIR

bazel build --config android --config optimization mace/libmace:libmace_static --define neon=true --define openmp=true --define opencl=true --cpu=$TARGET_ABI
cp -rf mace/public/*.h $INCLUDE_DIR
cp -rf bazel-genfiles/mace/libmace/libmace.a $LIBMACE_DIR

popd

if [ $# -eq 1 ] && [ $1 == "build" ]; then
    ./gradlew build
else
    ./gradlew installAppRelease
fi
