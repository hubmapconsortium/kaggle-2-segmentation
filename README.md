# hra-multiftu-segmentation-pipeline
Segmentation pipeline for multiple Functional Tissue Units (FTUs) based on code from the Kaggle #2 Competition.

Added support for WSI images.
Dockerized the inference script
Instructions for preparing docker file, building and running it can be found in the pdf files above

Packaging the K2 inference script docker
1. Base image:
a. Use a nvidia cuda image as a base image.
b. Ensure to use the devel version for further building the docker image on it.
c. The CUDA version that worked for pytorch 1.13 was 11.5
2. System packages:
a. Always update the apt-get package first
b. Then install the following packages in the same command:
i. libblas-dev
ii. liblapack-dev
iii. ffmpeg
iv. libsm6
v. libxext6
vi. gfortan
vii. git
viii. python3
ix. python3-dev
x. python3-pip
c. The first 2 packages are required for installing correctly and creating wheels for the
spams package.
d. System packages 3-6 are necessary for correct installation of opencv-python packages.
3. Python packages:
a. Install the pytorch package separately as it has a complex installation.
b. All the required packages are listed in the requirements.txt
4. Prediction files:
a. Copy the prediction files from the k2_inf folder into the container.
b. Model weights and data files are not present on github due to size constraints. (Can be downloaded from *get link from Yash*)
c. Model weights are in the .pb format and image data supported are: ome-tiff, tiff, png and jpeg.

Building and running the docker container
Steps to build the docker image:
1. Navigate into the docker folder
2. Build docker using following command:
docker build -f Dockerfile.v2 --network=host .
3. Keep the –netwrok=host parameter to enable internet connection inside the
docker container.
4. Find the docker image id using:
docker image ls
5. Run the docker container with the image_id using :
docker run --gpus all -dit “image_id”
6. Use –gpus parameter to dedicate gpus to the docker container and pass the -dit
parameter to let the docker container running until we exit it
7. Get the docker container id using:
docker ps -a
8. Attach the docker container to run prediction file:
docker attach “container_id”
9. Run the prediction file:
python3 predict_2.py


