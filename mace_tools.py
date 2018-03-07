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


def get_libs(target_abi, configs):
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

  libmace_name = "libmace-{}-{}".format(target_abi, global_runtime)

  command = "bash tools/download_and_link_lib.sh " + libmace_name
  run_command(command)

  return libmace_name


def clear_env():
  command = "bash tools/clear_env.sh"
  run_command(command)


def generate_random_input(model_output_dir):
  generate_data_or_not = True
  command = "bash tools/validate_tools.sh {} {}".format(
      model_output_dir, int(generate_data_or_not))
  run_command(command)


def generate_model_code():
  command = "bash tools/generate_model_code.sh"
  run_command(command)


def build_mace_run(production_mode, model_output_dir, hexagon_mode):
  command = "bash tools/build_mace_run.sh {} {} {}".format(
      int(production_mode), model_output_dir, int(hexagon_mode))
  run_command(command)


def tuning_run(model_output_dir, running_round, tuning, production_mode):
  command = "bash tools/tuning_run.sh {} {} {} {}".format(
      model_output_dir, running_round, int(tuning), int(production_mode))
  run_command(command)


def benchmark_model(model_output_dir):
  command = "bash tools/benchmark.sh {}".format(model_output_dir)
  run_command(command)


def run_model(model_output_dir, running_round):
  tuning_run(model_output_dir, running_round, False, False)


def generate_production_code(model_output_dirs, pull_or_not):
  cl_bin_dirs = []
  for d in model_output_dirs:
    cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
  cl_bin_dirs_str = ",".join(cl_bin_dirs)
  command = "bash tools/generate_production_code.sh {} {}".format(
      cl_bin_dirs_str, int(pull_or_not))
  run_command(command)


def build_mace_run_prod(model_output_dir, tuning, libmace_name):
  if "dsp" in libmace_name:
    hexagon_mode = True
  else:
    hexagon_mode = False

  production_or_not = False
  build_mace_run(production_or_not, model_output_dir, hexagon_mode)
  tuning_run(
      model_output_dir,
      running_round=0,
      tuning=tuning,
      production_mode=production_or_not)

  production_or_not = True
  pull_or_not = True
  generate_production_code([model_output_dir], pull_or_not)
  build_mace_run(production_or_not, model_output_dir, hexagon_mode)


def validate_model(model_output_dir):
  generate_data_or_not = False
  command = "bash tools/validate_tools.sh {} {}".format(
      model_output_dir, int(generate_data_or_not))
  run_command(command)


def build_production_code():
  command = "bash tools/build_production_code.sh"
  run_command(command)


def merge_libs_and_tuning_results(output_dir, model_output_dirs):
  pull_or_not = False
  generate_production_code(model_output_dirs, pull_or_not)
  build_production_code()

  model_output_dirs_str = ",".join(model_output_dirs)
  command = "bash tools/merge_libs.sh {} {}".format(output_dir,
                                                    model_output_dirs_str)
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
      "--tuning", type="bool", default="true", help="Tune opencl params.")
  parser.add_argument(
      "--mode", type=str, default="all", help="[build|run|validate|merge|all].")
  return parser.parse_known_args()


def main(unused_args):
  configs = parse_model_configs()

  if FLAGS.mode == "build" or FLAGS.mode == "all":
    # Remove previous output dirs
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    elif os.path.exists(os.path.join(FLAGS.output_dir, "libmace")):
      shutil.rmtree(os.path.join(FLAGS.output_dir, "libmace"))

  if FLAGS.mode == "validate":
    FLAGS.round = 1

  # target_abi = configs["target_abi"]
  # libmace_name = get_libs(target_abi, configs)
  # Transfer params by environment
  # os.environ["TARGET_ABI"] = target_abi
  os.environ["EMBED_MODEL_DATA"] = str(configs["embed_model_data"])
  os.environ["VLOG_LEVEL"] = str(configs["vlog_level"])
  os.environ["PROJECT_NAME"] = os.path.splitext(os.path.basename(FLAGS.config))[0]

  for target_abi in configs["target_abis"]:
    libmace_name = get_libs(target_abi, configs)
    # Transfer params by environment
    os.environ["TARGET_ABI"] = target_abi
    model_output_dirs = []
    for model_name in configs["models"]:
      # Transfer params by environment
      os.environ["MODEL_TAG"] = model_name
      model_config = configs["models"][model_name]
      for key in model_config:
        if key in ['input_node', 'output_node'] and isinstance(model_config[key], list):
            os.environ[key.upper()] = ",".join(model_config[key])
        elif key in ['input_shape', 'output_shape'] and isinstance(model_config[key], list):
            os.environ[key.upper()] = ":".join(model_config[key])
        else:
          os.environ[key.upper()] = str(model_config[key])

      md5 = hashlib.md5()
      md5.update(model_config["model_file_path"])
      model_path_digest = md5.hexdigest()
      model_output_dir = "%s/%s/%s/%s" % (FLAGS.output_dir, model_name, model_path_digest, target_abi)
      model_output_dirs.append(model_output_dir)

      if FLAGS.mode == "build" or FLAGS.mode == "all":
        if os.path.exists(model_output_dir):
          shutil.rmtree(model_output_dir)
        os.makedirs(model_output_dir)
        clear_env()

      # Support http:// and https://
      if model_config["model_file_path"].startswith(
          "http://") or model_config["model_file_path"].startswith("https://"):
        os.environ["MODEL_FILE_PATH"] = model_output_dir + "/model.pb"
        urllib.urlretrieve(model_config["model_file_path"], os.environ["MODEL_FILE_PATH"])

      if model_config["platform"] == "caffe" and (model_config["weight_file_path"].startswith(
          "http://") or model_config["weight_file_path"].startswith("https://")):
        os.environ["WEIGHT_FILE_PATH"] = model_output_dir + "/model.caffemodel"
        urllib.urlretrieve(model_config["weight_file_path"], os.environ["WEIGHT_FILE_PATH"])

      if FLAGS.mode == "build" or FLAGS.mode == "run" or FLAGS.mode == "validate" or FLAGS.mode == "all":
        generate_random_input(model_output_dir)

      if FLAGS.mode == "build" or FLAGS.mode == "all":
        generate_model_code()
        build_mace_run_prod(model_output_dir, FLAGS.tuning, libmace_name)

      if FLAGS.mode == "run" or FLAGS.mode == "validate" or FLAGS.mode == "all":
        run_model(model_output_dir, FLAGS.round)

      if FLAGS.mode == "benchmark":
        benchmark_model(model_output_dir)

      if FLAGS.mode == "validate" or FLAGS.mode == "all":
        validate_model(model_output_dir)

    if FLAGS.mode == "build" or FLAGS.mode == "merge" or FLAGS.mode == "all":
      merge_libs_and_tuning_results(FLAGS.output_dir + "/" + target_abi,
                                    model_output_dirs)


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
