#!/bin/bash

ehco "Note that this file has to be run from this directory currently for the relative paths to work"

echo "I can't use the debugger with this right now which makes it hard to use!.. I might want to look for another way to run locally and on AWS seamlessly..."

echo "I also don't think this can properly read local files with C:// because it's running on mnt/c/... This is a big problem"

echo "Starting training..."

python ../train_detector.py \
    --aws False \
    --aws_testing False \
    --run_validation False\
    --save_weights_when_testing True