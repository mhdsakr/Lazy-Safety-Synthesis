#!/bin/bash
dir=$(realpath $(dirname "$0"))
cd "$dir"
PYTHONPATH="$PYTHONPATH:$dir"
PATH="$PATH:$dir"
python2.7 "$dir/LazySafetySynt.py" "$@" 
