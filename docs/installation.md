## Installation

1. **Install Docker.**

   ```shell
   # Our project was tested in the official docker environment for reproducibility. (pytorch 1.8.0-cuda11.1-cudnn8-devel)

   $ sudo wget -qO- http://get.docker.com/ | sh
   
   $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   $ sudo apt-get update
   $ sudo apt-get install -y nvidia-docker2
   $ sudo systemctl restart docker
   ```

2. **Docker pull.**

   ```shell
   $ docker pull pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
   ```

3. **Clone this git-repository.**
   ```shell
   $ git clone https://github.com/kwakhyunmin/ENSEMBLE_TEACHER.git
   ```

4. **Docker run.**

   ```shell
   $ sudo docker run -it --gpus all --shm-size {Set the size of shared memory} --name ens_teacher -v {your directory(git clone repo)}:/workspace/ensemble_teacher pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel /bin/bash
   # For example:
   $ sudo docker run -it --gpus all --shm-size 16G --name ens_teacher -v /mnt/nas4/hm/semi_semantic/ensemble_teacher/ensemble_teacher:/workspace/ensemble_teacher pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel /bin/bash
   ```
   
5. **Install NVIDIA APEX.**

   ```shell
   $ apt-get update
   $ cd ensemble_teacher/apex/
   $ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   $ apt-get update -y
   $ apt-get install libgl1-mesa-glx
   $ apt-get install libglib2.0-0
   ```
  
6. **Install python libraries.**

   ```shell
   $ cd ..
   $ bash requirements.sh
   ```  
  
7. **Prepare dataset.**

   In our project, dataset (VOC, Cityscapes) and pre-trained models are constructed exactly like Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision
   ([CPS](https://arxiv.org/abs/2106.01226)) studies.
   Refer to the link and download it and configure it as below. (https://github.com/charlesCXK/TorchSemiSeg/blob/main/docs/getting_started.md)

   ```shell
   DATA/
   |-- city
   |-- pascal_voc
   |-- pytorch-weight
   |   |-- resnet50_v1c.pth
   |   |-- resnet101_v1c.pth
   ```  
  
  
  
  
  
  
  
  
  
 
