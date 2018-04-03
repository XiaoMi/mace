import sh
import re
import time
import falcon_cli

################################
# common
################################
def strip_invalid_utf8(str):
  return sh.iconv(str, "-c", "-t", "UTF-8")

def make_output_processor(buff):
  def process_output(line):
    print(line.strip())
    buff.append(line)
  return process_output

################################
# adb commands
################################
def adb_split_stdout(stdout_str):
  stdout_str = strip_invalid_utf8(stdout_str)
  # Filter out last empty line
  return [l.strip() for l in stdout_str.split('\n') if len(l.strip()) > 0]

def adb_devices(target_socs=None):
  outputs = sh.grep(sh.adb("devices"), "^[A-Za-z0-9]\+[[:space:]]\+device$")
  raw_lists = sh.cut(outputs, "-f1")
  device_ids = adb_split_stdout(raw_lists)
  if target_socs != None:
    target_socs_set = set(target_socs)
    target_devices = []
    for serialno in device_ids:
      props = adb_getprop_by_serialno(serialno)
      if props["ro.board.platform"] in target_socs_set:
        target_devices.append(serialno)
    return target_devices
  else:
    return device_ids

def adb_getprop_by_serialno(serialno):
  outputs = sh.adb("-s", serialno, "shell", "getprop")
  raw_props = adb_split_stdout(outputs)
  props = {}
  p = re.compile("\[(.+)\]: \[(.+)\]")
  for raw_prop in raw_props:
    m = p.match(raw_prop)
    if m:
      props[m.group(1)] = m.group(2)
  return props

def adb_supported_abis(serialno):
  props = adb_getprop_by_serialno(serialno)
  abilist_str = props["ro.product.cpu.abilist"]
  abis = [abi.strip() for abi in abilist_str.split(',')]
  return abis

def adb_get_all_socs():
  socs = []
  for d in adb_devices():
    props = adb_getprop_by_serialno(d)
    socs.append(props["ro.board.platform"])
  return set(socs)

def adb_run(serialno, host_bin_path, bin_name,
            args="",
            opencl_profiling=1,
            vlog_level=0,
            device_bin_path="/data/local/tmp/mace"):
  host_bin_full_path = "%s/%s" % (host_bin_path, bin_name)
  device_bin_full_path = "%s/%s" % (device_bin_path, bin_name)
  device_cl_path = "%s/cl" % device_bin_path
  props = adb_getprop_by_serialno(serialno)
  print("=====================================================================")
  print("Run on device: %s, %s, %s" % (serialno, props["ro.board.platform"],
                                       props["ro.product.model"]))
  sh.adb("-s", serialno, "shell", "rm -rf %s" % device_bin_path)
  sh.adb("-s", serialno, "shell", "mkdir -p %s" % device_bin_path)
  sh.adb("-s", serialno, "shell", "mkdir -p %s" % device_cl_path)
  print("Push %s to %s" % (host_bin_full_path, device_bin_full_path))
  sh.adb("-s", serialno, "push", host_bin_full_path, device_bin_path)
  print("Run %s" % device_bin_full_path)
  stdout_buff=[]
  process_output = make_output_processor(stdout_buff)
  p = sh.adb("-s", serialno, "shell",
             "MACE_OPENCL_PROFILING=%d MACE_KERNEL_PATH=%s MACE_CPP_MIN_VLOG_LEVEL=%d %s %s" %
             (opencl_profiling, device_cl_path, vlog_level, device_bin_full_path, args),
             _out=process_output, _bg=True, _err_to_out=True)
  p.wait()
  return "".join(stdout_buff)


################################
# bazel commands
################################
def bazel_build(target, strip="always", abi="armeabi-v7a"):
  print("Build %s with ABI %s" % (target, abi))
  stdout_buff=[]
  process_output = make_output_processor(stdout_buff)
  p= sh.bazel("build",
              "-c", "opt",
              "--strip", strip,
              "--verbose_failures",
              target,
              "--crosstool_top=//external:android/crosstool",
              "--host_crosstool_top=@bazel_tools//tools/cpp:toolchain",
              "--cpu=%s" % abi,
              "--copt=-std=c++11",
              "--copt=-D_GLIBCXX_USE_C99_MATH_TR1",
              "--copt=-DMACE_DISABLE_NO_TUNING_WARNING",
              "--copt=-Werror=return-type",
              "--copt=-O3",
              "--define", "neon=true",
              "--define", "openmp=true",
              _out=process_output, _bg=True, _err_to_out=True)
  p.wait()
  return "".join(stdout_buff)

def bazel_target_to_bin(target):
  # change //mace/a/b:c to bazel-bin/mace/a/b/c
  prefix, bin_name = target.split(':')
  prefix = prefix.replace('//', '/')
  if prefix.startswith('/'):
    prefix = prefix[1:]
  host_bin_path = "bazel-bin/%s" % prefix
  return host_bin_path, bin_name

################################
# mace commands
################################
# TODO this should be refactored
def gen_encrypted_opencl_source(codegen_path="mace/codegen"):
  sh.python("mace/python/tools/encrypt_opencl_codegen.py",
            "--cl_kernel_dir=./mace/kernels/opencl/cl/",
            "--output_path=%s/opencl/opencl_encrypt_program.cc" % codegen_path)

def gen_mace_version(codegen_path="mace/codegen"):
  sh.mkdir("-p", "%s/version" % codegen_path)
  sh.bash("mace/tools/git/gen_version_source.sh",
          "%s/version/version.cc" % codegen_path)

################################
# falcon
################################
def falcon_tags(platform, model, abi):
  return "ro.board.platform=%s,ro.product.model=%s,abi=%s" % (platform, model, abi)

def falcon_push_metrics(metrics, device_properties, abi, endpoint="mace_dev"):
  cli = falcon_cli.FalconCli.connect(server="transfer.falcon.miliao.srv",
                                     port=8433,
                                     debug=False)
  platform = device_properties["ro.board.platform"].replace(" ", "-")
  model = device_properties["ro.product.model"].replace(" ", "-")
  tags = falcon_tags(platform, model, abi)
  ts = int(time.time())
  falcon_metrics = [{
      "endpoint": endpoint,
      "metric": key,
      "tags": tags,
      "timestamp": ts,
      "value": value,
      "step": 86400,
      "counterType": "GAUGE"
      } for key, value in metrics.iteritems()]
  cli.update(falcon_metrics)

