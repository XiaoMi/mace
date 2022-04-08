#!/bin/bash
# Copyright 2020 The MACE Authors. All Rights Reserved.

set -e

LIB_DIR=build/lib
INCLUDE_DIR=build/include

# colors for terminal display
declare -r RED='\033[0;31m'
declare -r NC='\033[0m'        # No Color
declare -r BOLD=$(tput bold)
declare -r NORMAL=$(tput sgr0)

# helper function
helper() {
  echo -e "usage:\t$0 ["${BOLD}"--abi"${NORMAL}"=abi]\
["${BOLD}"--runtimes"${NORMAL}"=rt1,rt2,...]["${BOLD}"--static"${NORMAL}"]"

  echo -e "\t"${BOLD}"--abi:"${NORMAL}" specifies the targeted ABI, supported \
ABIs are:\n\t\tarmeabi-v7a, arm64-v8a, arm_linux_gnueabihf, aarch64_linux_gnu \
or \n\t\thost if the library is built for the host machine (linux-x86-64).\n\t\
\tThe default ABI is arm64-v8a."

  echo -e "\t"${BOLD}"--runtimes:"${NORMAL}" specifies the runtimes, supported \
runtimes are:\n\t\tcpu, gpu, dsp, apu, hta, htp. By default, the library is \
built to run on CPU."

  echo -e "\t"${BOLD}"--static:"${NORMAL}" option to generate the corresponding\
 static library.\n\t\tIf the option is omitted, a shared library is built."

  exit 0
}

# default configuration variables
abi=arm64-v8a
enable_neon=false
enable_hta=false
enable_cpu=false
enable_gpu=false
enable_dsp=false
enable_apu=false
enable_htp=false
enable_qnn=false
enable_quantize=false
enable_bfloat16=false
enable_rpcmem=true
enable_debug=false
enable_asan=false
static_lib=false
symbol_hidden=
runtime_label=""
lib_type=dynamic
lib_label=shared
lib_suffix=.so
bazel_dir=bazel-bin

# make lib and include directories
mkdir -p "${LIB_DIR}"
mkdir -p "${INCLUDE_DIR}"

# copy include headers
cp -R include/mace ${INCLUDE_DIR}/

# positional parameters parsing
for opt in "${@}";do
  case "${opt}" in
    abi=*|-abi=*|--abi=*)
      abi="$(echo "${opt}" | cut -d '=' -f 2-)"
      ;;
    runtime=*|runtimes=*|-runtime=*|-runtimes=*|--runtime=*|--runtimes=*)
      arg="$(echo "$opt" | cut -d '=' -f 2-)"
      for runtime in $(echo $arg | sed s/,/\ /g);do
        case $runtime in
          cpu|CPU)
            enable_cpu=true
            enable_neon=true
            runtime_label=""${runtime_label}" cpu"
            ;;
          gpu|GPU)
            enable_cpu=true
            enable_neon=true
            enable_gpu=true
            runtime_label=""${runtime_label}" gpu"
            ;;
          dsp|DSP)
            enable_dsp=true
            runtime_label=""${runtime_label}" dsp"
            ;;
          apu|APU)
            enable_apu=true
            runtime_label=""${runtime_label}" apu"
            ;;
          hta|HTA)
            enable_hta=true
            runtime_label=""${runtime_label}" hta"
            ;;
          htp|HTP)
            enable_htp=true
            enable_qnn=true
            runtime_label=""${runtime_label}" htp"
            ;;
          *)
            echo -e ""${RED}""${BOLD}"ERROR:"${NORMAL}""${NC}" unknown device \
"${device}""
            exit 1
            ;;
        esac
      done
      ;;
    static|-static|--static)
      static_lib=true
      ;;
    debug|-debug|--debug)
      enable_debug=true
      ;;
    asan|-asan|--asan)
      enable_asan=true
      ;;
    quantize|-quantize|--quantize)
      enable_quantize=true
      ;;
    bfloat16|-bfloat16|--bfloat16)
      enable_bfloat16=true
      ;;
    help|-help|--help)
      helper
      ;;
    *)
      echo -e ""${RED}""${BOLD}"ERROR:"${NORMAL}""${NC}" unknown parameter \
$(echo "$1" | cut -d '=' -f -1)"
      echo -e "See \x27$0 --help\x27 for more information"
      exit 1
      ;;
  esac
done

if [[ "${enable_apu}" == true || ("${abi}" != armeabi-v7a && "${abi}" != arm64-v8a) ]];then
  enable_rpcmem=false
fi

if [[ "${static_lib}" == true ]];then
  lib_type=static
  lib_label=static
  lib_suffix=.a
  symbol_hidden="--config symbol_hidden"
  bazel_dir=bazel-genfiles
fi

debug_str=""
if [[ "${enable_debug}" == true ]];then
  debug_str="debug"
fi

asan_str=""
if [[ "${enable_asan}" == true ]];then
  asan_str="asan"
fi

if [[ "${abi}" == host || "${abi}" == HOST ]];then
  abi=linux-x86-64
fi

# make the directory
rm -rf "${LIB_DIR}"/"${abi}"
mkdir "${LIB_DIR}"/"${abi}"
# build the target library
build_msg="build "${lib_label}" lib for "${abi}" "${debug_str}" "${asan_str}""
if [[ "${abi}" != linux-x86-64 ]];then
  build_msg=""${build_msg}" + "${runtime_label}""
fi
echo "-------------${build_msg}------------"
case "${abi}" in
  arm_linux_gnueabihf|aarch64_linux_gnu)
    bazel build --config "${abi}" --config "${debug_str}" --config "${asan_str}" \
    --config optimization mace/libmace:libmace_"${lib_type}" \
    --define cpu_enabled="${enable_cpu}" \
    --define neon="${enable_neon}" \
    --define opencl="${enable_gpu}" \
    --define quantize="${enable_quantize}" \
    --define bfloat16="${enable_bfloat16}"
    ;;
  linux-x86-64)
    bazel build mace/libmace:libmace_"${lib_type}" --config linux \
    --config "${debug_str}" --config "${asan_str}" --config optimization \
    cp "${bazel_dir}"/mace/libmace/libmace"${lib_suffix}" "${LIB_DIR}"/"${abi}"/
    ;;
  *)
    apu_version="-1"
    if [[ "${enable_apu}" == true ]];then
        set +e
        python tools/python/apu_utils.py get-version --target_abi=${abi}
        apu_version=`echo $?`
        echo "The apu version is ${apu_version}"
        set -e
    fi
    bazel build --config android --config "${debug_str}" \
    --config "${asan_str}" --config optimization \
    mace/libmace:libmace_"${lib_type}" ${symbol_hidden} \
    --define cpu_enabled="${enable_cpu}" \
    --define neon="${enable_neon}" --define hta="${enable_hta}" \
    --define opencl="${enable_gpu}" --define apu="${enable_apu}" \
    --define hexagon="${enable_dsp}" --define quantize="${enable_quantize}" \
    --define rpcmem="${enable_rpcmem}" --define bfloat16="${enable_bfloat16}" \
    --cpu="${abi}" --define apu_version="${apu_version}" \
    --define qnn="${enable_qnn}"
    if [[ "${enable_dsp}" == true ]];then
      cp third_party/nnlib/"${abi}"/libhexagon_controller.so \
      "${LIB_DIR}"/"${abi}"/
    fi
    if [[ "${enable_apu}" == true ]];then
      # Detect the plugin-device and copy the valid so to the output dir
      python tools/python/apu_utils.py copy-so-files --target_abi ${abi} --apu_path "${LIB_DIR}"/"${abi}"/
    fi
    if [[ "${enable_hta}" == true ]];then
      cp third_party/hta/"${abi}"/*so "${LIB_DIR}"/"${abi}"/
    fi
    if [[ "${enable_qnn}" == true ]];then
      if [[ "${abi}" == "arm64-v8a" ]];then
        cp third_party/qnn/target/aarch64-android/lib/*so "${LIB_DIR}"/"${abi}"/
      fi
      if [[ "${abi}" == "armeabi-v7a" ]];then
        cp third_party/qnn/target/arm-android/lib/*so "${LIB_DIR}"/"${abi}"/
      fi
    fi
    ;;
esac
cp "${bazel_dir}"/mace/libmace/libmace"${lib_suffix}" "${LIB_DIR}"/"${abi}"/

echo "LIB PATH: ${LIB_DIR}"
echo "INCLUDE FILE PATH: ${INCLUDE_DIR}"
