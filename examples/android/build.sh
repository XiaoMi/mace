#!/usr/bin/env bash

set -e -u -o pipefail

Usage() {
  echo "Usage: ./build.sh [dynamic|static]"
  echo "|==============|====================|"
  echo "|   parameter  |  lib will linked   |"
  echo "|==============|====================|"
  echo "|   dynamic    |    libmace.so      |"
  echo "|--------------|--------------------|"
  echo "|   static     |    libmace.a       |"
  echo "|--------------|--------------------|"
}

if [ $# -lt 1 ]; then
  Usage
  exit 1
fi

MACE_LINK_TYPE=$1

pushd ../..

TARGET_ABI=arm64-v8a
ANDROID_DEMO_DIR=examples/android/
LIBRARY_DIR=$ANDROID_DEMO_DIR/macelibrary/src/main/cpp/
INCLUDE_DIR=$LIBRARY_DIR/include
LIBMACE_DIR=$LIBRARY_DIR/lib/$TARGET_ABI/
LIBGNUSTL_SHARED_SO=libgnustl_shared.so
LIBCPP_SHARED_SO=libc++_shared.so

JNILIBS_DIR=$ANDROID_DEMO_DIR/macelibrary/src/main/jniLibs/$TARGET_ABI
rm -rf $JNILIBS_DIR

if [ $MACE_LINK_TYPE == "dynamic" ]; then
  BAZEL_LIBMACE_TARGET=mace/libmace:libmace.so
  BAZEL_GEN_LIBMACE_PATH=bazel-bin/mace/libmace/libmace.so
elif [ $MACE_LINK_TYPE == "static" ]; then
  BAZEL_LIBMACE_TARGET=mace/libmace:libmace_static
  BAZEL_GEN_LIBMACE_PATH=bazel-genfiles/mace/libmace/libmace.a
else
  Usage
  exit 1
fi

python tools/converter.py convert --config=examples/android/mobilenet.yml --target_abis=$TARGET_ABI

rm -rf $INCLUDE_DIR && mkdir -p $INCLUDE_DIR
rm -rf $LIBMACE_DIR && mkdir -p $LIBMACE_DIR
rm -rf $LIBRARY_DIR/model/

cp -rf include/mace $INCLUDE_DIR
cp -rf build/mobilenet/include/mace/public/*.h $INCLUDE_DIR/mace/public/
cp -rf build/mobilenet/model $LIBRARY_DIR

bazel build --config android --config optimization $BAZEL_LIBMACE_TARGET --define neon=true --define openmp=true --define opencl=true --define quantize=true --cpu=$TARGET_ABI
cp -rf $BAZEL_GEN_LIBMACE_PATH $LIBMACE_DIR

if [ $MACE_LINK_TYPE == "dynamic" ]; then
  mkdir -p $JNILIBS_DIR
  cp -rf $BAZEL_GEN_LIBMACE_PATH $JNILIBS_DIR

  if [[ "" != `$ANDROID_NDK_HOME/ndk-depends $BAZEL_GEN_LIBMACE_PATH | grep $LIBGNUSTL_SHARED_SO` ]]; then
    cp -rf $ANDROID_NDK_HOME/sources/cxx-stl/gnu-libstdc++/4.9/libs/$TARGET_ABI/$LIBGNUSTL_SHARED_SO $JNILIBS_DIR
  fi

  if [[ "" != `$ANDROID_NDK_HOME/ndk-depends $BAZEL_GEN_LIBMACE_PATH | grep $LIBCPP_SHARED_SO` ]]; then
    cp -rf $ANDROID_NDK_HOME/sources/cxx-stl/llvm-libc++/libs/$TARGET_ABI/$LIBCPP_SHARED_SO $JNILIBS_DIR
  fi
fi

popd

# Build demo
./gradlew clean
./gradlew build
