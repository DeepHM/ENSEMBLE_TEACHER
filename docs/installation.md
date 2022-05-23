## Installation

1. **Clone this repo.**

   ```shell
   $ sudo wget -qO- http://get.docker.com/ | sh
   $ cd TorchSemiSeg
   ```

  Our project was tested in the official docker environment for reproducibility. (pytorch 1.8.0-cuda11.1-cudnn8-devel)

   ```shell
   $ sudo wget -qO- http://get.docker.com/ | sh
   $ cd TorchSemiSeg
   ```

2. **Clone this repo.**

   ```shell
   $ git clone https://github.com/charlesCXK/TorchSemiSeg.git
   $ cd TorchSemiSeg
   ```

3. **Install dependencies.**

   **(1) Create a conda environment:**

   ```shell
   $ conda env create -f semiseg.yaml
   $ conda activate semiseg
   ```

   **(2) Install apex 0.1(needs CUDA)**

   ```shell
   $ cd ./furnace/apex
   $ python setup.py install --cpp_ext --cuda_ext
   ```
  
## Optional
We recommend using docker to run experiments. Here is the docker name: charlescxk/ssc:2.0 .
You could pull it from https://hub.docker.com/ and mount your local file system to it.
