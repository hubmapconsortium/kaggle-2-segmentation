#!/bin/bash
docker build -f docker/Dockerfile.v2 --network=host -t multiftu .
docker run --rm --user=$(id -u):$(id -g) -v $PWD/data:/data -v $PWD/output:/output -v $PWD/weights:/opt/weights --gpus all -it multiftu python3 inference.py --data_directory /data/k1_kidney_test_set --output_directory /output/k1_kidney_test_set --tissue_type kidney --inference_mode fast