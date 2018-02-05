#!/usr/bin/env python

# Must run at root dir of libmace project.
# python tools/mace_tools.py \
#     --global_config=models/config \
#     --round=100 \
#     --mode=all

import argparse
import os
import shutil
import subprocess
import sys

from ConfigParser import ConfigParser

tf_model_file_dir_key = "TF_MODEL_FILE_DIR"


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


def get_libs(configs):
  libmace_name = "libmace"
  for config in configs:
    if config["ANDROID_ABI"] == "armeabi-v7a":
      libmace_name += "_v7"
      break
    elif config["ANDROID_ABI"] == "arm64-v8a":
      libmace_name += "_v8"
      break

  for config in configs:
    if config["RUNTIME"] == "dsp":
      libmace_name += "_dsp"
      break
    if config["RUNTIME"] == "local":
      libmace_name += "_local"
      break

  command = "bash tools/download_and_link_lib.sh " + libmace_name
  run_command(command)


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


def build_mace_run(production_mode, model_output_dir):
  command = "bash tools/build_mace_run.sh {} {}".format(
      int(production_mode), model_output_dir)
  run_command(command)


def tuning_run(model_output_dir, running_round, tuning, production_mode):
  command = "bash tools/tuning_run.sh {} {} {} {}".format(
      model_output_dir, running_round, int(tuning), int(production_mode))
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


def build_mace_run_prod(model_output_dir, tuning):
  production_or_not = False
  build_mace_run(production_or_not, model_output_dir)
  tuning_run(
      model_output_dir,
      running_round=0,
      tuning=tuning,
      production_mode=production_or_not)

  production_or_not = True
  pull_or_not = True
  generate_production_code([model_output_dir], pull_or_not)
  build_mace_run(production_or_not, model_output_dir)


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
  production_or_not = True
  build_production_code()

  model_output_dirs_str = ",".join(model_output_dirs)
  command = "bash tools/merge_libs.sh {} {}".format(output_dir,
                                                    model_output_dirs_str)
  run_command(command)


def parse_sub_model_configs(model_dirs, global_configs):
  model_configs = []
  for model_dir in model_dirs:
    model_config = {}

    model_config_path = os.path.join(model_dir, "config")
    if os.path.exists(model_config_path):
      cf = ConfigParser()
      # Preserve character case
      cf.optionxform = str
      cf.read(model_config_path)
      if "configs" in cf.sections():
        config_list = cf.items("configs")
        for config_map in config_list:
          model_config[config_map[0]] = config_map[1]
      else:
        raise Exception("No config msg found in {}".format(model_config_path))
    else:
      raise Exception("Config file '{}' not found".format(model_config_path))

    model_config[tf_model_file_dir_key] = model_dir

    for config_map in global_configs:
      model_config[config_map[0]] = config_map[1]

    model_configs.append(model_config)

  return model_configs


def parse_model_configs():
  config_parser = ConfigParser()
  # Preserve character case
  config_parser.optionxform = str

  global_config_dir = os.path.dirname(FLAGS.global_config)

  try:
    config_parser.read(FLAGS.global_config)
    config_sections = config_parser.sections()

    model_dirs = []
    model_output_map = {}
    if ("models" in config_sections) and (config_parser.items("models")):
      model_dirs_str = config_parser.get(
          "models", "DIRECTORIES")
      model_dirs_str = model_dirs_str.rstrip(
          ",")

      # Remove repetition element
      model_dirs = list(
          set(model_dirs_str.split(",")))

      for model_dir in model_dirs:
        # Create output dirs
        model_output_dir = os.path.join(FLAGS.output_dir, model_dir)

        model_output_map[model_dir] = model_output_dir
    else:
      model_dirs = [global_config_dir]

      # Create output dirs
      model_output_dir = os.path.join(FLAGS.output_dir, global_config_dir)
      model_output_map[global_config_dir] = model_output_dir
  except Exception as e:
    print("Error in read model path msg. Exception: {}".format(e))
    return

  global_configs = []
  if "configs" in config_sections:
    global_configs = config_parser.items("configs")

  return parse_sub_model_configs(model_dirs, global_configs), model_output_map


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--global_config",
      type=str,
      default="./tool/config",
      help="The global config file of models.")
  parser.add_argument(
      "--output_dir", type=str, default="./build/", help="The output dir.")
  parser.add_argument(
      "--round", type=int, default=1, help="The model running round.")
  parser.add_argument(
      "--tuning", type="bool", default="true", help="Tune opencl params.")
  parser.add_argument(
      "--mode", type=str, default="all", help="[build|run|merge|all].")
  return parser.parse_known_args()


def main(unused_args):
  configs, model_output_map = parse_model_configs()

  if FLAGS.mode == "build" or FLAGS.mode == "all":
    # Remove previous output dirs
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    elif os.path.exists(os.path.join(FLAGS.output_dir, "libmace")):
      shutil.rmtree(os.path.join(FLAGS.output_dir, "libmace"))

  get_libs(configs)

  if FLAGS.mode == "run" and len(configs) > 1:
    raise Exception("Mode 'run' only can execute one model config, which have been built lastest")

  model_output_dirs = []
  for config in configs:
    # Transfer params by environment
    for key in config:
      os.environ[key] = config[key]
    model_output_dir = model_output_map[config[tf_model_file_dir_key]]
    model_output_dirs.append(model_output_dir)

    if FLAGS.mode == "build" or FLAGS.mode == "all":
      if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
      os.makedirs(model_output_dir)
      clear_env()

    if FLAGS.mode == "build" or FLAGS.mode == "run" or FLAGS.mode == "all":
      generate_random_input(model_output_dir)

    if FLAGS.mode == "build" or FLAGS.mode == "all":
      generate_model_code()
      build_mace_run_prod(model_output_dir, FLAGS.tuning)

    if FLAGS.mode == "run" or FLAGS.mode == "all":
      run_model(model_output_dir, FLAGS.round)

    if FLAGS.mode == "all":
      validate_model(model_output_dir)

  if FLAGS.mode == "build" or FLAGS.mode == "merge" or FLAGS.mode == "all":
    merge_libs_and_tuning_results(FLAGS.output_dir, model_output_dirs)


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
