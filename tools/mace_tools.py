#!/usr/bin/env python

# Must run at root dir of libmace project.
# python tools/mace_tools.py \
#     --config=tools/example.yaml \
#     --round=100 \
#     --mode=all

import argparse
import hashlib
import os
import sh
import shutil
import subprocess
import sys
import urllib
import yaml
import re

import sh_commands

from ConfigParser import ConfigParser


def run_command(command):
  print("Run command: {}".format(command))
  result = subprocess.Popen(
      command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = result.communicate()

  if out:
    print("Stdout msg:\n{}".format(out))
  if err:
    print("Stderr msg:\n{}".format(err))

  if result.returncode != 0:
    raise Exception("Exit not 0 from bash with code: {}, command: {}".format(
        result.returncode, command))


def get_global_runtime(configs):
  runtime_list = []
  for model_name in configs["models"]:
    model_runtime = configs["models"][model_name]["runtime"]
    runtime_list.append(model_runtime.lower())

  global_runtime = ""
  if "dsp" in runtime_list:
    global_runtime = "dsp"
  elif "gpu" in runtime_list:
    global_runtime = "gpu"
  elif "cpu" in runtime_list:
    global_runtime = "cpu"
  elif "neon" in runtime_list:
    global_runtime = "neon"
  else:
    raise Exception("Not found available RUNTIME in config files!")

  return global_runtime


def generate_version_code():
  command = "bash tools/generate_version_code.sh"
  run_command(command)

def generate_opencl_source_code():
  command = "bash tools/generate_opencl_code.sh source"
  run_command(command)

def generate_opencl_binay_code(target_soc, model_output_dirs, pull_or_not):
  cl_bin_dirs = []
  for d in model_output_dirs:
    cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
  cl_bin_dirs_str = ",".join(cl_bin_dirs)
  if not cl_bin_dirs:
    command = "bash tools/generate_opencl_code.sh binary"
  else:
    command = "bash tools/generate_opencl_code.sh {} {} {} {}".format(
      'binary', target_soc, cl_bin_dirs_str, int(pull_or_not))
  run_command(command)

def generate_tuning_param_code(target_soc, model_output_dirs, pull_or_not):
  cl_bin_dirs = []
  for d in model_output_dirs:
    cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
  cl_bin_dirs_str = ",".join(cl_bin_dirs)
  if not cl_bin_dirs:
    command = "bash tools/generate_tuning_param_code.sh"
  else:
    command = "bash tools/generate_tuning_param_code.sh {} {} {}".format(
      target_soc, cl_bin_dirs_str, int(pull_or_not))
  run_command(command)

def generate_code(target_soc, model_output_dirs, pull_or_not):
  generate_opencl_binay_code(target_soc, model_output_dirs, pull_or_not)
  generate_tuning_param_code(target_soc, model_output_dirs, pull_or_not)

def clear_env(target_soc):
  command = "bash tools/clear_env.sh {}".format(target_soc)
  run_command(command)

def input_file_name(input_name):
  return os.environ['INPUT_FILE_NAME'] + '_' + \
         re.sub('[^0-9a-zA-Z]+', '_', input_name)

def generate_random_input(target_soc, model_output_dir,
                          input_names, input_files):
  generate_data_or_not = True
  command = "bash tools/validate_tools.sh {} {} {}".format(
      target_soc, model_output_dir, int(generate_data_or_not))
  run_command(command)

  input_file_list = []
  if isinstance(input_files, list):
    input_file_list.extend(input_files)
  else:
    input_file_list.append(input_files)
  if len(input_file_list) != 0:
    input_name_list = []
    if isinstance(input_names, list):
      input_name_list.extend(input_names)
    else:
      input_name_list.append(input_names)
    if len(input_file_list) != len(input_name_list):
      raise Exception('If input_files set, the input files should match the input names.')
    for i in range(len(input_file_list)):
      if input_file_list[i] is not None:
        dst_input_file = model_output_dir + '/' + input_file_name(input_name_list[i])
        if input_file_list[i].startswith("http://") or \
            input_file_list[i].startswith("https://"):
          urllib.urlretrieve(input_file_list[i], dst_input_file)
        else:
          shutil.copy(input_file_list[i], dst_input_file)

def generate_model_code():
  command = "bash tools/generate_model_code.sh"
  run_command(command)


def build_mace_run(production_mode, model_output_dir, hexagon_mode):
  command = "bash tools/build_mace_run.sh {} {} {}".format(
      int(production_mode), model_output_dir, int(hexagon_mode))
  run_command(command)


def tuning_run(model_name,
               target_runtime,
               target_abi,
               target_soc,
               model_output_dir,
               running_round,
               tuning,
               restart_round,
               option_args=''):
  # TODO(yejianwu) refactoring the hackish code
  stdout_buff = []
  process_output = sh_commands.make_output_processor(stdout_buff)
  p = sh.bash("tools/tuning_run.sh", target_soc, model_output_dir,
              running_round, int(tuning), int(production_mode),
              restart_round, option_args, _out=process_output,
              _bg=True, _err_to_out=True)
  p.wait()
  metrics = {}
  for line in stdout_buff:
    line = line.strip()
    parts = line.split()
    if len(parts) == 6 and parts[0].startswith("time"):
      metrics["%s.create_net_ms" % model_name] = str(float(parts[1]))
      metrics["%s.mace_engine_ctor_ms" % model_name] = str(float(parts[2]))
      metrics["%s.init_ms" % model_name] = str(float(parts[3]))
      metrics["%s.warmup_ms" % model_name] = str(float(parts[4]))
      if float(parts[5]) > 0:
        metrics["%s.avg_latency_ms" % model_name] = str(float(parts[5]))
  tags = {"ro.board.platform": target_soc,
          "abi": target_abi,
          # "runtime": target_runtime, # TODO(yejianwu) Add the actual runtime
          "round": running_round, # TODO(yejianwu) change this to source/binary
          "tuning": tuning}
  sh_commands.falcon_push_metrics(metrics, endpoint="mace_model_benchmark",
                                  tags=tags)

def benchmark_model(target_soc, model_output_dir, option_args=''):
  command = "bash tools/benchmark.sh {} {} \"{}\"".format(
      target_soc, model_output_dir, option_args)
  run_command(command)


def run_model(model_name, target_runtime, target_abi, target_soc,
              model_output_dir, running_round, restart_round, option_args):
  tuning_run(model_name, target_runtime, target_abi, target_soc,
             model_output_dir, running_round, False, False,
             restart_round, option_args)


def generate_production_code(target_soc, model_output_dirs, pull_or_not):
  cl_bin_dirs = []
  for d in model_output_dirs:
    cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
  cl_bin_dirs_str = ",".join(cl_bin_dirs)
  command = "bash tools/generate_production_code.sh {} {} {}".format(
      target_soc, cl_bin_dirs_str, int(pull_or_not))
  run_command(command)


def build_mace_run_prod(model_name, target_runtime, target_abi, target_soc,
                        model_output_dir, tuning):
  if "dsp" == target_runtime:
    hexagon_mode = True
  else:
    hexagon_mode = False

  generate_code(target_soc, [], False)
  production_or_not = False
  build_mace_run(production_or_not, model_output_dir, hexagon_mode)
  tuning_run(
      model_name,
      target_runtime,
      target_abi, 
      target_soc,
      model_output_dir,
      running_round=0,
      tuning=tuning,
      restart_round=1)

  generate_code(target_soc, [model_output_dir], True)
  production_or_not = True
  build_mace_run(production_or_not, model_output_dir, hexagon_mode)


def build_run_throughput_test(target_soc, run_seconds, merged_lib_file,
                              model_input_dir):
  command = "bash tools/build_run_throughput_test.sh {} {} {} {}".format(
      target_soc, run_seconds, merged_lib_file, model_input_dir)
  run_command(command)


def validate_model(target_soc, model_output_dir):
  generate_data_or_not = False
  command = "bash tools/validate_tools.sh {} {} {}".format(
      target_soc, model_output_dir, int(generate_data_or_not))
  run_command(command)


def build_production_code():
  command = "bash tools/build_production_code.sh"
  run_command(command)


def merge_libs_and_tuning_results(target_soc, output_dir, model_output_dirs):
  generate_code(target_soc, model_output_dirs, False)
  build_production_code()

  model_output_dirs_str = ",".join(model_output_dirs)
  command = "bash tools/merge_libs.sh {} {} {}".format(target_soc, output_dir,
                                                       model_output_dirs_str)
  run_command(command)


def packaging_lib_file(output_dir):
  command = "bash tools/packaging_lib.sh {}".format(output_dir)
  run_command(command)

def download_model_files(model_file_path,
                         model_output_dir,
                         weight_file_path=""):
  if model_file_path.startswith("http://") or \
      model_file_path.startswith("https://"):
    os.environ["MODEL_FILE_PATH"] = model_output_dir + "/model.pb"
    urllib.urlretrieve(model_file_path, os.environ["MODEL_FILE_PATH"])

  if weight_file_path.startswith("http://") or \
      weight_file_path.startswith("https://"):
    os.environ[
      "WEIGHT_FILE_PATH"] = model_output_dir + "/model.caffemodel"
    urllib.urlretrieve(weight_file_path,
      os.environ["WEIGHT_FILE_PATH"])

def md5sum(str):
  md5 = hashlib.md5()
  md5.update(str)
  return md5.hexdigest()


def parse_model_configs():
  with open(FLAGS.config) as f:
    configs = yaml.load(f)
    return configs


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--config",
      type=str,
      default="./tool/config",
      help="The global config file of models.")
  parser.add_argument(
      "--output_dir", type=str, default="build", help="The output dir.")
  parser.add_argument(
      "--round", type=int, default=1, help="The model running round.")
  parser.add_argument(
      "--run_seconds",
      type=int,
      default=10,
      help="The model throughput test running seconds.")
  parser.add_argument(
      "--restart_round", type=int, default=1, help="The model restart round.")
  parser.add_argument(
      "--tuning", type="bool", default="true", help="Tune opencl params.")
  parser.add_argument(
      "--mode",
      type=str,
      default="all",
      help="[build|run|validate|merge|all|throughput_test].")
  parser.add_argument(
      "--target_socs",
      type=str,
      default="all",
      help="SoCs to build, comma seperated list (getprop ro.board.platform)")
  return parser.parse_known_args()

def set_environment(configs):
  os.environ["EMBED_MODEL_DATA"] = str(configs["embed_model_data"])
  os.environ["VLOG_LEVEL"] = str(configs["vlog_level"])
  os.environ["PROJECT_NAME"] = os.path.splitext(os.path.basename(
    FLAGS.config))[0]
  os.environ['INPUT_FILE_NAME'] = "model_input"
  os.environ['OUTPUT_FILE_NAME'] = "model_out"

def main(unused_args):
  configs = parse_model_configs()

  if FLAGS.mode == "validate":
    FLAGS.round = 1
    FLAGS.restart_round = 1

  set_environment(configs)

  if FLAGS.mode == "build" or FLAGS.mode == "all":
    # Remove previous output dirs
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    elif os.path.exists(os.path.join(FLAGS.output_dir, "libmace")):
      shutil.rmtree(os.path.join(FLAGS.output_dir, os.environ["PROJECT_NAME"]))
      os.makedirs(os.path.join(FLAGS.output_dir, os.environ["PROJECT_NAME"]))

    generate_version_code()
    generate_opencl_source_code()

  option_args = ' '.join([arg for arg in unused_args if arg.startswith('--')])

  available_socs = sh_commands.adb_get_all_socs()
  target_socs = available_socs
  if hasattr(configs, "target_socs"):
    target_socs = set(configs["target_socs"])
    target_socs = target_socs & available_socs

  if FLAGS.target_socs != "all":
    socs = set(FLAGS.target_socs.split(','))
    target_socs = target_socs & socs
    missing_socs = socs.difference(target_socs)
    if len(missing_socs) > 0:
      print("Error: devices with SoCs are not connected %s" % missing_socs)
      exit(1)


  for target_soc in target_socs:
    for target_abi in configs["target_abis"]:
      global_runtime = get_global_runtime(configs)
      # Transfer params by environment
      os.environ["TARGET_ABI"] = target_abi
      model_output_dirs = []
      for model_name in configs["models"]:
        print '=======================', model_name, '======================='
        # Transfer params by environment
        os.environ["MODEL_TAG"] = model_name
        model_config = configs["models"][model_name]
        input_file_list = model_config.get("validation_inputs_data", [])
        for key in model_config:
          if key in ['input_nodes', 'output_nodes'] and isinstance(
              model_config[key], list):
            os.environ[key.upper()] = ",".join(model_config[key])
          elif key in ['input_shapes', 'output_shapes'] and isinstance(
              model_config[key], list):
            os.environ[key.upper()] = ":".join(model_config[key])
          else:
            os.environ[key.upper()] = str(model_config[key])

        # Create model build directory
        model_path_digest = md5sum(model_config["model_file_path"])
        model_output_dir = "%s/%s/%s/%s/%s/%s/%s" % (FLAGS.output_dir,
                                                     os.environ["PROJECT_NAME"],
                                                     "build", model_name,
                                                     model_path_digest,
                                                     target_soc, target_abi)
        model_output_dirs.append(model_output_dir)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
          if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
          os.makedirs(model_output_dir)
          clear_env(target_soc)

        download_model_files(model_config["model_file_path"],
          model_output_dir, model_config.get("weight_file_path", ""))

        if FLAGS.mode == "build" or FLAGS.mode == "run" or FLAGS.mode == "validate"\
            or FLAGS.mode == "benchmark" or FLAGS.mode == "all":
          generate_random_input(target_soc, model_output_dir,
            model_config['input_nodes'], input_file_list)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
          generate_model_code()
          build_mace_run_prod(model_name, global_runtime, target_abi,
                              target_soc, model_output_dir, FLAGS.tuning)

        if FLAGS.mode == "run" or FLAGS.mode == "validate" or FLAGS.mode == "all":
          run_model(model_name, global_runtime, target_abi, target_soc,
                    model_output_dir, FLAGS.round, FLAGS.restart_round,
                    option_args)

        if FLAGS.mode == "benchmark":
          benchmark_model(target_soc, model_output_dir, option_args)

        if FLAGS.mode == "validate" or FLAGS.mode == "all":
          validate_model(target_soc, model_output_dir)

      if FLAGS.mode == "build" or FLAGS.mode == "merge" or FLAGS.mode == "all":
        merge_libs_and_tuning_results(
            target_soc, FLAGS.output_dir + "/" + os.environ["PROJECT_NAME"],
            model_output_dirs)

      if FLAGS.mode == "throughput_test":
        merged_lib_file = FLAGS.output_dir + "/%s/%s/libmace_%s.%s.a" % \
            (os.environ["PROJECT_NAME"], target_abi, os.environ["PROJECT_NAME"], target_soc)
        generate_random_input(target_soc, FLAGS.output_dir, [], [])
        for model_name in configs["models"]:
          runtime = configs["models"][model_name]["runtime"]
          os.environ["%s_MODEL_TAG" % runtime.upper()] = model_name
        build_run_throughput_test(target_soc, FLAGS.run_seconds,
                                  merged_lib_file, FLAGS.output_dir)

  if FLAGS.mode == "build" or FLAGS.mode == "all":
    packaging_lib_file(FLAGS.output_dir)


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

