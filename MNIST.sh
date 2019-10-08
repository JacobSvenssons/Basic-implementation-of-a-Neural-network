#!/bin/bash

# Script for executing the Python version of digits
# Usage:
# bash MNIST.sh <training-images.txt> <training-label.txt> <validation-images.txt>

# Author: Ola Ringdahl

# the location of this script:
base_dir="$(dirname "$0")"

# Call your program with the input to the script
python3 $base_dir/digits.py $1 $2 $3

