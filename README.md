# YOLO-NCNN-Deployment-Guide

run docker
```
docker run -it --gpus all -v $(pwd):/home container_id
```

install dependencies
```bash
apt update && apt upgrade -y
apt install python3 python3-pip -y
apt install build-essential git cmake wget libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev -y
```
## Build NCNN
clone NCNN source code
```bash
cd work_dir
git clone https://github.com/Tencent/ncnn.git
```
```bash
cd ncnn
mkdir build && cd build
cmake ..
make -j16
make install
```
run demo, verified installation
```bash
cd ../examples
../build/examples/squeezenet ../images/256-ncnn.png
```
If show below results, then installation success
```
532 = 0.165649
920 = 0.094421
716 = 0.062408
```

## YOLOv5

### 1. Install PyTorch and Dependencies

*Note: Adjust the PyTorch install command according to your device and CUDA version.*

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas opencv-python-headless pyyaml tqdm matplotlib seaborn onnx onnxsim protobuf
```

---

### 2. Download YOLOv5 `.pt` Weights

```bash
# Clone YOLOv5 repository and checkout v7.0
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout v7.0

# Install requirements
pip install -r requirements.txt --user

# Download pretrained weights
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

---

### 3. Export Model to TorchScript

```bash
python export.py --weights yolov5s.pt --include torchscript
# Output file: yolov5s.torchscript
```

---

### 4. Convert TorchScript to NCNN Using pnnx

```bash
# Download latest pnnx release
wget https://github.com/pnnx/pnnx/releases/download/20250530/pnnx-20250530-linux.zip
unzip pnnx-20250530-linux.zip

# Convert TorchScript model to NCNN format
./pnnx-20250530-linux/pnnx yolov5s.torchscript inputshape=[1,3,640,640]
# Output files: yolov5s.ncnn.param, yolov5s.ncnn.bin
```

---

### 5. Copy NCNN Model Files

```bash
cd work_dir
mkdir -p models
cd models

cp /home/yolov5/yolov5s.ncnn.param .
cp /home/yolov5/yolov5s.ncnn.bin .
```

---

## YOLOv7

### Converting YOLOv7 `.pt` Model to NCNN Format

#### 1. Clone YOLOv7 Repository

```bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
```

#### 2. Download Pretrained Weights

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

#### 3. Export to TorchScript

```bash
python models/export.py --weights yolov7.pt
# Output file: yolov7.torchscript.pt
```

#### 4. Convert TorchScript to NCNN Using pnnx

```bash
./pnnx-20250530-linux/pnnx yolov7.torchscript.pt inputshape=[1,3,640,640]
# Output files:
# - yolov7.torchscript.ncnn.param
# - yolov7.torchscript.ncnn.bin
```

---

## Building Your Own YOLOv5 Project with NCNN

### 1. Create `CMakeLists.txt`

```cmake
project(yolov5)

# TODO: Change to your NCNN installation path
set(ncnn_DIR "/home/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(yolov5 yolov5.cpp)
target_link_libraries(yolov5 ncnn ${OpenCV_LIBS})
```

---

### 2. Build Project

```bash
cd my-ncnn
mkdir -p build
cd build

cmake ..
make -j16
```

---

### 3. Run Inference

```bash
cd bin
./yolov5 ../test.jpg
```

---
# Directory Tree
```bash
root@0af71fa1fde7:/home# tree -L 2
.
|-- CMakeLists.txt
|-- README.md
|-- bin
|   `-- yolov5
|-- build
|   |-- CMakeCache.txt
|   |-- CMakeFiles
|   |-- Makefile
|   `-- cmake_install.cmake
|-- models
|   |-- yolov5s.ncnn.bin
|   `-- yolov5s.ncnn.param
|-- ncnn
|   |-- CITATION.cff
|   |-- CMakeLists.txt
|   |-- CONTRIBUTING.md
|   |-- Info.plist
|   |-- LICENSE.txt
|   |-- MANIFEST.in
|   |-- README.md
|   |-- benchmark
|   |-- build
|   |-- build-android.cmd
|   |-- build.sh
|   |-- cmake
|   |-- codeformat.sh
|   |-- docs
|   |-- examples
|   |-- glslang
|   |-- images
|   |-- package.sh
|   |-- pyproject.toml
|   |-- python
|   |-- setup.py
|   |-- src
|   |-- tests
|   |-- toolchains
|   `-- tools
|-- test.jpg
|-- yolov5
|   |-- CONTRIBUTING.md
|   |-- LICENSE
|   |-- README.md
|   |-- benchmarks.py
|   |-- classify
|   |-- data
|   |-- debug.bin
|   |-- debug.param
|   |-- debug2.bin
|   |-- debug2.param
|   |-- detect.py
|   |-- export.py
|   |-- hubconf.py
|   |-- models
|   |-- pnnx-20230217-ubuntu
|   |-- pnnx-20230217-ubuntu.zip
|   |-- pnnx-20250403-linux
|   |-- pnnx-20250403-linux.zip
|   |-- requirements.txt
|   |-- segment
|   |-- setup.cfg
|   |-- train.py
|   |-- tutorial.ipynb
|   |-- utils
|   |-- val.py
|   |-- yolov5s.ncnn.bin
|   |-- yolov5s.ncnn.param
|   |-- yolov5s.pnnx.bin
|   |-- yolov5s.pnnx.onnx
|   |-- yolov5s.pnnx.param
|   |-- yolov5s.pt
|   |-- yolov5s.torchscript
|   |-- yolov5s_ncnn.py
|   `-- yolov5s_pnnx.py
`-- yolov5.cpp
```



# Error handling
```bash
E: Failed to fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/./libxnvctrl0_575.57.08-0ubuntu1_amd64.deb  File has unexpected size (11948 != 11944). Mirror sync in progress? [IP: 203.66.199.32 443]
   Hashes of expected file:
    - SHA512:78f7552b4f3d0de14bfe817cb4fd671ab7196126be827731c17b9c3fcb5a744546fc9d2d33b0566d57e38fab46218bd824ebbf8666f41686ac1a024f2d7851c7
    - SHA256:82e3ad4f54080f3f3d5dad8b30a9eec11ef02e2bcea3ec7502e90a23b3157ae9
    - SHA1:567a337d97493a3457d6f24a0566f5c266545aa1 [weak]
    - MD5Sum:48d304c161b5c8b58f0ae2e0a8cc8356 [weak]
    - Filesize:11944 [weak]
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
```
```bash
rm /etc/apt/sources.list.d/cuda.list
```
