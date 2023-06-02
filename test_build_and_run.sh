#!/bin/bash
docker build -f docker/Dockerfile.v2 --network=host -t multiftu .
docker run -v $PWD/data:/data -v $PWD/output:/output -v $PWD/weights:/opt/weights --gpus all -it multiftu python3 inference.py --inference_mode fast