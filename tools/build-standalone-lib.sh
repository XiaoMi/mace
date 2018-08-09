#!/bin/bash

set -e

LIB_DIR=builds/lib
INCLUDE_DIR=builds/include/mace/public

mkdir -p $LIB_DIR
mkdir -p $INCLUDE_DIR

# copy include headers
cp mace/public/*.h $INCLUDE_DIR/

# make directories
rm -rf $LIB_DIR/armeabi-v7a
mkdir -p $LIB_DIR/armeabi-v7a/cpu_gpu_dsp
mkdir -p $LIB_DIR/armeabi-v7a/cpu_gpu

rm -rf $LIB_DIR/arm64-v8a
mkdir -p $LIB_DIR/arm64-v8a/cpu_gpu

rm -rf $LIB_DIR/linux-x86-64
mkdir -p $LIB_DIR/linux-x86-64

# build shared libraries
echo "build shared lib for armeabi-v7a + cpu_gpu_dsp"
bazel build --config android --config optimization mace/libmace:libmace_dynamic --define neon=true --define openmp=true --define opencl=true --define hexagon=true --cpu=armeabi-v7a
cp bazel-bin/mace/libmace/libmace.so $LIB_DIR/armeabi-v7a/cpu_gpu_dsp/
cp third_party/nnlib/*so $LIB_DIR/armeabi-v7a/cpu_gpu_dsp/

echo "build shared lib for armeabi-v7a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_dynamic --define neon=true --define openmp=true --define opencl=true --cpu=armeabi-v7a
cp bazel-bin/mace/libmace/libmace.so $LIB_DIR/armeabi-v7a/cpu_gpu/

echo "build shared lib for arm64-v8a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_dynamic --define neon=true --define openmp=true --define opencl=true --cpu=arm64-v8a
cp bazel-bin/mace/libmace/libmace.so $LIB_DIR/arm64-v8a/cpu_gpu/

if [[ "$OSTYPE" != "darwin"* ]];then
	echo "build shared lib for linux-x86-64"
	bazel build mace/libmace:libmace_dynamic --config optimization --define openmp=true
	cp bazel-bin/mace/libmace/libmace.so $LIB_DIR/linux-x86-64/
fi

# build static libraries
echo "build static lib for armeabi-v7a + cpu_gpu_dsp"
bazel build --config android --config optimization mace/libmace:libmace_static --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --define hexagon=true --cpu=armeabi-v7a
cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/armeabi-v7a/cpu_gpu_dsp/
cp third_party/nnlib/*so $LIB_DIR/armeabi-v7a/cpu_gpu_dsp/

echo "build static lib for armeabi-v7a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_static --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --cpu=armeabi-v7a
cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/armeabi-v7a/cpu_gpu/

echo "build static lib for arm64-v8a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_static --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --cpu=arm64-v8a
cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/arm64-v8a/cpu_gpu/

if [[ "$OSTYPE" != "darwin"* ]];then
	echo "build static lib for linux-x86-64"
	bazel build mace/libmace:libmace_static --config optimization --define openmp=true
	cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/linux-x86-64/
fi

echo "LIB PATH: $LIB_DIR"
echo "INCLUDE FILE PATH: $INCLUDE_DIR"
