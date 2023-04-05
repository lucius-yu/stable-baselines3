#!/bin/bash


export CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=index,memory.free, --format=csv,noheader | sort -nr -k 2 | awk -F ',' 'NR<3 {print $1}' | sed -z 's/\n/,/g;s/,$/\n/'`
