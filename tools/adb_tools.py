import sh
import re

def adb_split_stdout(stdout_str):
  # Filter out last empty line
  return [l.strip() for l in stdout_str.split('\n') if len(l.strip()) > 0]

def adb_devices():
  outputs = sh.grep(sh.adb("devices"), "^[A-Za-z0-9]\+[[:space:]]\+device$")
  raw_lists = sh.cut(outputs, "-f1")
  return adb_split_stdout(raw_lists)

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

def adb_get_all_socs():
  socs = []
  for d in adb_devices():
    props = adb_getprop_by_serialno(d)
    socs.append(props["ro.product.board"])
  return set(socs)
