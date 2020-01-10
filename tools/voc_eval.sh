#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON $(dirname "$0")/test.py $1 $2 --out $3

$PYTHON $(dirname "$0")/voc_eval.py $3 $1 ${@:4}