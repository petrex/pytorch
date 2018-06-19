mport shutil
import subprocess
import os
import sys
from shutil import copytree, ignore_patterns
from functools import reduce
from os.path import basename

proj_dir = "~/pytorch"
ignore_files = []
hipify = "/opt/rocm/bin/hipify-perl"


# HIPCC Compiler doesn't provide host defines - Automatically include them.
for root, _, files in os.walk(os.path.join(proj_dir, "caffe2")):
    for filename in files:
    if filename.endswith(".cu") or filename.endswith(".cuh") and ("cudnn" not in filename) and (filename not in ignore_files):
        filepath = os.path.join(root, filename)
# run hipify
        hipName = basename(filename)+"_hip.cc"
        subprocess.Popen([hipify, filepath, '>', 'hip/', hipName])


