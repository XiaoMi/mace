#!/usr/bin/env python

# Must run at root dir of libmace project.
# python tools/mace_tools.py \
#     --config=tools/example.yaml \
#     --round=100 \
#     --mode=all

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib
import yaml

import adb_tools

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
  else:
    raise Exception("Not found available RUNTIME in config files!")

  return global_runtime


def generate_opencl_and_version_code():
  command = "bash tools/generate_opencl_and_version_code.sh"
  run_command(command)


def clear_env(target_soc):
  command = "bash tools/clear_env.sh {}".format(target_soc)
  run_command(command)


def generate_random_input(target_soc, model_output_dir):
  generate_data_or_not = True
  command = "bash tools/validate_tools.sh {} {} {}".format(
      target_soc, model_output_dir, int(generate_data_or_not))
  run_command(command)


def generate_model_code():
  command = "bash tools/generate_model_code.sh"
  run_command(command)


def build_mace_run(production_mode, model_output_dir, hexagon_mode):
  command = "bash tools/build_mace_run.sh {} {} {}".format(
      int(production_mode), model_output_dir, int(hexagon_mode))
  run_command(command)


def tuning_run(target_soc,
               model_output_dir,
               running_round,
               tuning,
               production_mode,
               restart_round,
               option_args=''):
  command = "bash tools/tuning_run.sh {} {} {} {} {} {} \"{}\"".format(
      target_soc, model_output_dir, running_round, int(tuning),
      int(production_mode), restart_round, option_args)
  run_command(command)


def benchmark_model(target_soc, model_output_dir, option_args=''):
  command = "bash tools/benchmark.sh {} {} \"{}\"".format(
      target_soc, model_output_dir, option_args)
  run_command(command)


def run_model(target_soc, model_output_dir, running_round, restart_round,
              option_args):
  tuning_run(target_soc, model_output_dir, running_round, False, False,
             restart_round, option_args)


def generate_production_code(target_soc, model_output_dirs, pull_or_not):
  cl_bin_dirs = []
  for d in model_output_dirs:
    cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
  cl_bin_dirs_str = ",".join(cl_bin_dirs)
  command = "bash tools/generate_production_code.sh {} {} {}".format(
      target_soc, cl_bin_dirs_str, int(pull_or_not))
  run_command(command)


def build_mace_run_prod(target_soc, model_output_dir, tuning, global_runtime):
  if "dsp" == global_runtime:
    hexagon_mode = True
  else:
    hexagon_mode = False

  production_or_not = False
  build_mace_run(production_or_not, model_output_dir, hexagon_mode)
  tuning_run(
      target_soc,
      model_output_dir,
      running_round=0,
      tuning=tuning,
      production_mode=production_or_not,
      restart_round=1)

  production_or_not = True
  pull_or_not = True
  generate_production_code(target_soc, [model_output_dir], pull_or_not)
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
  pull_or_not = False
  generate_production_code(target_soc, model_output_dirs, pull_or_not)
  build_production_code()

  model_output_dirs_str = ",".join(model_output_dirs)
  command = "bash tools/merge_libs.sh {} {} {}".format(target_soc, output_dir,
                                                       model_output_dirs_str)
  run_command(command)

def packaging_lib_file(output_dir):
  command = "bash tools/packaging_lib.sh {}".format(output_dir)
  run_command(command)


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
      "--socs",
      type=str,
      default="all",
      help="SoCs to build, comma seperated list (getprop ro.board.platform)")
  return parser.parse_known_args()


def main(unused_args):
  configs = parse_model_configs()

  if FLAGS.mode == "validate":
    FLAGS.round = 1
    FLAGS.restart_round = 1

  os.environ["EMBED_MODEL_DATA"] = str(configs["embed_model_data"])
  os.environ["VLOG_LEVEL"] = str(configs["vlog_level"])
  os.environ["PROJECT_NAME"] = os.path.splitext(os.path.basename(
      FLAGS.config))[0]

  if FLAGS.mode == "build" or FLAGS.mode == "all":
    # Remove previous output dirs
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    elif os.path.exists(os.path.join(FLAGS.output_dir, "libmace")):
      shutil.rmtree(os.path.join(FLAGS.output_dir, os.environ["PROJECT_NAME"]))
      os.makedirs(os.path.join(FLAGS.output_dir, os.environ["PROJECT_NAME"]))

  generate_opencl_and_version_code()
  option_args = ' '.join([arg for arg in unused_args if arg.startswith('--')])

  available_socs = adb_tools.adb_get_all_socs()
  target_socs = available_socs
  if hasattr(configs, "target_socs"):
    target_socs = set(configs["target_socs"])
    target_socs = target_socs & available_socs

  if FLAGS.socs != "all":
    socs = set(FLAGS.socs.split(','))
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
        # Transfer params by environment
        os.environ["MODEL_TAG"] = model_name
        print '=======================', model_name, '======================='
        skip_validation = configs["models"][model_name].get("skip_validation", 0)
        model_config = configs["models"][model_name]
        for key in model_config:
          if key in ['input_nodes', 'output_nodes'] and isinstance(
              model_config[key], list):
            os.environ[key.upper()] = ",".join(model_config[key])
          elif key in ['input_shapes', 'output_shapes'] and isinstance(
              model_config[key], list):
            os.environ[key.upper()] = ":".join(model_config[key])
          else:
            os.environ[key.upper()] = str(model_config[key])

        md5 = hashlib.md5()
        md5.update(model_config["model_file_path"])
        model_path_digest = md5.hexdigest()
        model_output_dir = "%s/%s/%s/%s/%s/%s/%s" % (FLAGS.output_dir, os.environ["PROJECT_NAME"],
                                               "build", model_name,
                                               model_path_digest, target_soc,
                                               target_abi)
        model_output_dirs.append(model_output_dir)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
          if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
          os.makedirs(model_output_dir)
          clear_env(target_soc)

        # Support http:// and https://
        if model_config["model_file_path"].startswith(
            "http://") or model_config["model_file_path"].startswith(
                "https://"):
          os.environ["MODEL_FILE_PATH"] = model_output_dir + "/model.pb"
          urllib.urlretrieve(model_config["model_file_path"],
                             os.environ["MODEL_FILE_PATH"])

        if model_config["platform"] == "caffe" and (
            model_config["weight_file_path"].startswith("http://") or
            model_config["weight_file_path"].startswith("https://")):
          os.environ[
              "WEIGHT_FILE_PATH"] = model_output_dir + "/model.caffemodel"
          urllib.urlretrieve(model_config["weight_file_path"],
                             os.environ["WEIGHT_FILE_PATH"])

        if FLAGS.mode == "build" or FLAGS.mode == "run" or FLAGS.mode == "validate"\
            or FLAGS.mode == "benchmark" or FLAGS.mode == "all":
          generate_random_input(target_soc, model_output_dir)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
          generate_model_code()
          build_mace_run_prod(target_soc, model_output_dir, FLAGS.tuning,
                              global_runtime)

        if FLAGS.mode == "run" or FLAGS.mode == "validate" or FLAGS.mode == "all":
          run_model(target_soc, model_output_dir, FLAGS.round,
                    FLAGS.restart_round, option_args)

        if FLAGS.mode == "benchmark":
          benchmark_model(target_soc, model_output_dir, option_args)

        if FLAGS.mode == "validate" or (FLAGS.mode == "all" and skip_validation == 0):
          validate_model(target_soc, model_output_dir)

      if FLAGS.mode == "build" or FLAGS.mode == "merge" or FLAGS.mode == "all":
        merge_libs_and_tuning_results(
            target_soc, FLAGS.output_dir + "/" + os.environ["PROJECT_NAME"],
            model_output_dirs)

      if FLAGS.mode == "throughput_test":
        merged_lib_file = FLAGS.output_dir + "/%s/%s/libmace_%s.%s.a" % \
            (os.environ["PROJECT_NAME"], target_abi, os.environ["PROJECT_NAME"], target_soc)
        generate_random_input(target_soc, FLAGS.output_dir)
        for model_name in configs["models"]:
          runtime = configs["models"][model_name]["runtime"]
          os.environ["%s_MODEL_TAG" % runtime.upper()] = model_name
        build_run_throughput_test(target_soc, FLAGS.run_seconds,
                                  merged_lib_file, FLAGS.output_dir)

  packaging_lib_file(FLAGS.output_dir)


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
