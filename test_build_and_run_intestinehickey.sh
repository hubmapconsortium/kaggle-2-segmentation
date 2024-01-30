#!/bin/bash
docker build -f docker/Dockerfile.v2 --network=host -t multiftu .
docker run -v --user=$(id -u):$(id -g) $PWD/data:/data -v $PWD/output:/output -v $PWD/weights:/opt/weights --gpus all -it multiftu python3 inference.py --data_directory /data/intestine_hickey --output_directory /output/intestine_hickey --tissue_type largeintestine --inference_mode fast