Here’s a cleaned-up, well-structured, and easier-to-read version of your README:

---

# YOLO-NCNN Deployment Guide

---

## Table of Contents

* [Run Docker Container](#run-docker-container)
* [Install Dependencies](#install-dependencies)
* [Build NCNN](#build-ncnn)
* [YOLOv5 Setup](#yolov5-setup)
* [YOLOv7 Setup](#yolov7-setup)
* [Build Your Own YOLO Project with NCNN](#build-your-own-yolo-project-with-ncnn)
* [Directory Structure](#directory-structure)
* [Error Handling](#error-handling)

---

## Run Docker Container

```bash
docker run -it --gpus all -v $(pwd):/home container_id
```

---

## Install Dependencies

```bash
apt update && apt upgrade -y
apt install python3 python3-pip -y
apt install build-essential git cmake wget libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev -y
```

---

## Build NCNN

### Clone NCNN Source

```bash
cd work_dir
git clone https://github.com/Tencent/ncnn.git
```

### Build and Install NCNN

```bash
cd ncnn
mkdir build && cd build
cmake ..
make -j16
make install
```

### Verify Installation by Running Demo

```bash
cd ../examples
../build/examples/squeezenet ../images/256-ncnn.png
```

If you see output similar to:

```
532 = 0.165649
920 = 0.094421
716 = 0.062408
```

Then the installation was successful.

---

## YOLOv5 Setup

### 1. Install PyTorch and Dependencies

*Adjust the PyTorch version for your CUDA version if needed.*

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas opencv-python-headless pyyaml tqdm matplotlib seaborn onnx onnxsim protobuf
```

---

### 2. Download YOLOv5 `.pt` Weights

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout v7.0
pip install -r requirements.txt --user
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

---

### 3. Export Model to TorchScript

```bash
python export.py --weights yolov5s.pt --include torchscript
# Generates yolov5s.torchscript
```

---

### 4. Convert TorchScript to NCNN Using pnnx

```bash
wget https://github.com/pnnx/pnnx/releases/download/20250530/pnnx-20250530-linux.zip
unzip pnnx-20250530-linux.zip
./pnnx-20250530-linux/pnnx yolov5s.torchscript inputshape=[1,3,640,640]
# Outputs yolov5s.ncnn.param and yolov5s.ncnn.bin
```

---

### 5. Copy NCNN Model Files to Project

```bash
cd work_dir
mkdir -p models
cd models

cp /home/yolov5/yolov5s.ncnn.param .
cp /home/yolov5/yolov5s.ncnn.bin .
```

---

## YOLOv7 Setup

### 1. Clone YOLOv7 Repository

```bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
```

### 2. Download Pretrained Weights

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### 3. Export to TorchScript

```bash
python models/export.py --weights yolov7.pt
# Generates yolov7.torchscript.pt
```

### 4. Convert TorchScript to NCNN Using pnnx

```bash
./pnnx-20250530-linux/pnnx yolov7.torchscript.pt inputshape=[1,3,640,640]
# Outputs:
# yolov7.torchscript.ncnn.param
# yolov7.torchscript.ncnn.bin
```
---

## Build Your Own YOLO Project with NCNN

### 1. Create `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn)

# Change this to your NCNN installation path
set(ncnn_DIR "/home/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add executables for YOLOv5 and YOLOv7 inference programs
add_executable(yolov5 yolov5.cpp)
target_link_libraries(yolov5 ncnn ${OpenCV_LIBS})

add_executable(yolov7 yolov7.cpp)
target_link_libraries(yolov7 ncnn ${OpenCV_LIBS})
```

---

### 2. Build the Project

```bash
cd your_project_dir
mkdir -p build
cd build

cmake ..
make -j16
```

---

### 3. Run Inference

```bash
# Run YOLOv5 inference
./bin/yolov5 ../test.jpg

# Run YOLOv7 inference
./bin/yolov7 ../test.jpg
```

## Directory Structure

```
root@0af71fa1fde7:/home# tree -L 2
.
|-- CMakeLists.txt
|-- README.md
|-- models
|   |-- yolov5s.ncnn.bin
|   |-- yolov5s.ncnn.param
|   |-- yolov7.torchscript.ncnn.bin
|   `-- yolov7.torchscript.ncnn.param
|-- ncnn
|-- pnnx-20250403-linux
|   |-- README.md
|   `-- pnnx
|-- test.jpg
|-- yolov5
|-- yolov5.cpp
|-- yolov7
`-- yolov7.cpp
```

---

## Error Handling

If you encounter errors like:

```bash
E: Failed to fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/./libxnvctrl0_575.57.08-0ubuntu1_amd64.deb  File has unexpected size (11948 != 11944). Mirror sync in progress? [IP: 203.66.199.32 443]
...
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
```

Try this fix:

```bash
rm /etc/apt/sources.list.d/cuda.list
```
