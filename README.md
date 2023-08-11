# hra-multiftu-segmentation-pipeline
Segmentation pipeline for multiple organs and Functional Tissue Units (FTUs) based on code from the Kaggle #2 Competition.
Supported organs: Kidney, Large Intestine, Lung, Prostate, Spleen. 

**Building and running the docker container**

Steps to build the docker image:

1. Navigate into `hra-multiftu-segmentation-pipeline` directory.
2. Build docker using following command: `docker build -f docker/Dockerfile.v2 --network=host -t multiftu .`
3. Keep the `–netwrok=host` parameter to enable internet connection inside the
docker container. `multiftu` is the name we have defined for the container.
4. You can check if docker image was successfully created using: `docker image ls`
5. Run the docker container using : `docker run --rm --user=$(id -u):$(id -g) -v $PWD/data:/data -v $PWD/output:/output -v $PWD/weights:/opt/weights --gpus all -it multiftu python3 inference.py --data_directory /data --output_directory /output --tissue_type kidney --inference_mode fast`. Optionally, give a `--config_file path_to_config.json`.
6. `$PWD/data:/data` mounts a local directory containing your input image. `$PWD/output:/output` mounts a local directory to the container where the output will be saved. [Optional for testing: `$PWD/weights:/opt/weights` mounts a local directory containing your model weights, see point 10 below.] 
7. You can also use this command to enter the container: `docker run --gpus all -it multiftu /bin/bash`
8. Use `–-gpus` parameter to dedicate gpus to the docker container.
9. You can check docker container status by using: `docker ps -a`
10. Trained model weights can be downloaded from: (https://zenodo.org/record/7996245). For testing, download weights and save in a dir `weights` in the root directory of the repository. Use `test_build_and_run.sh` for building the container and running the inference script. For production, uncomment the last line that's currently commented in dockerfile which will download all weights to the model during docker build.

**Details regarding containerizing the model into docker**
1. Base image:
    a. Use a nvidia cuda image as a base image.
    b. Ensure to use the devel version for further building the docker image on it.
    c. The CUDA version that worked for pytorch 1.13 was 11.5
2. System packages:
    a. Always update the apt-get package first
    b. Then install the following packages in the same command:
  
        libblas-dev
        liblapack-dev
        ffmpeg
        libsm6
        libxext6
        gfortan
        git
        python3
        python3-dev
        python3-pip
    
   c. The first 2 packages are required for installing correctly and creating wheels for the
     spams package.
   d. System packages 3-6 are necessary for correct installation of opencv-python packages.
3. Python packages:
    a. Install the pytorch package separately as it has a complex installation.
    b. All the required packages are listed in the requirements.txt
4. Prediction files:
    a. Copy the prediction script files from the bin folder into the container.
    b. Model weights and data files are not present on github due to size constraints. (Can be downloaded from *https://zenodo.org/record/7996245*)
    c. Model weights are in the .pb format and image data supported are: ome-tiff, tif, tiff.
    d. The model input locations and output locations along with other configurations can be accessed and changed via the conifg.json file
