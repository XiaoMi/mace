#!/usr/bin/env sh

set -e

# build for local host
sh tools/cmake-build-host.sh

# Nuild for Android arm64-v8a with NEON
sh tools/cmake-build-android-arm64-v8a-cpu.sh

# Build for Android arm64-v8a with NEON, Quantize, OpenCL, Hexagon HTA, Hexagon DSP
sh tools/cmake-build-android-arm64-v8a-full.sh

# Nuild for Android armeabi-v7a with NEON
sh tools/cmake-build-android-armeabi-v7a-cpu.sh

# Build for Android armeabi-v7a with NEON, Quantize, OpenCL, Hexagon HTA, Hexagon DSP
sh tools/cmake-build-android-armeabi-v7a-full.sh

# Build for arm-linux-gnueabihf with NEON, Quantize, OpenCL
sh tools/cmake-build-arm-linux-gnueabihf-full.sh

# Build for aarch64-linux-gnu with NEON, Quantize, OpenCL
sh tools/cmake-build-aarch64-linux-gnu-full.sh
