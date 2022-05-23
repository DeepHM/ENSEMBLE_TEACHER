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

3. ** Clone this repository.**
   ```shell
   $ git clone https://github.com/kwakhyunmin/ENSEMBLE_TEACHER.git
   ```

5. **Docker run.**

   ```shell
   $ sudo docker run -it --gpus all --shm-size {Set the size of shared memory} --name ens_teacher -v {your directory(git clone repo)}:/workspace/ensemble_teacher pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel /bin/bash
   # For example:
   $ sudo docker run -it --gpus all --shm-size 16G --name ens_teacher -v /mnt/nas4/hm/semi_semantic/ensemble_teacher/ensemble_teacher:/workspace/ensemble_teacher pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel /bin/bash
   ```
   
4. **Install NVIDIA APEX.**

   ```shell
   $ cd ensemble_teacher/apex/
   $ python setup.py install --cpp_ext --cuda_ext
   ```
  
  
## Optional
We recommend using docker to run experiments. Here is the docker name: charlescxk/ssc:2.0 .
You could pull it from https://hub.docker.com/ and mount your local file system to it.
