#!/bin/bash
docker build -f docker/Dockerfile.v2 --network=host -t multiftu .
docker run -v $PWD/data:/data -v $PWD/output:/output --gpus all -it multiftu python3 inference.py --inference_mode fast