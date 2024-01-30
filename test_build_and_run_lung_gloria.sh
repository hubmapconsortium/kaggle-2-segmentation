#!/bin/bash
docker build -f docker/Dockerfile.v2 --network=host -t multiftu .
docker run --user=$(id -u):$(id -g) -v $PWD/data:/data -v $PWD/output:/output -v $PWD/weights:/opt/weights --gpus all -it multiftu python3 inference.py --data_directory /data/lung_gloria --output_directory /output/lung_gloria --tissue_type lung --inference_mode fast