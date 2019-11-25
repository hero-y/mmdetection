#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo $(dirname "$0")
echo $0 #表示当前
echo $PYTHON
echo ${@:3}
echo ${@}