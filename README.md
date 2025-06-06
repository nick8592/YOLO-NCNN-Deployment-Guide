# my-ncnn

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

## YOLOv5-6.0 onnx to ncnn
### Install pytorch
depends on your device
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas opencv-python-headless pyyaml tqdm matplotlib seaborn onnx onnxsim protobuf
```

### Export onnx model
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 export.py --weights yolov5s.pt --include torchscript onnx
python3 -m onnxsim yolov5s.onnx yolov5s-sim.onnx
```
```
Simplifying...
Finish! Here is the difference:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add        │ 10             │ 10               │
│ Concat     │ 17             │ 17               │
│ Constant   │ 147            │ 138              │
│ Conv       │ 60             │ 60               │
│ MaxPool    │ 3              │ 3                │
│ Mul        │ 69             │ 69               │
│ Pow        │ 3              │ 3                │
│ Reshape    │ 6              │ 6                │
│ Resize     │ 2              │ 2                │
│ Sigmoid    │ 60             │ 60               │
│ Split      │ 3              │ 3                │
│ Transpose  │ 3              │ 3                │
│ Model Size │ 28.0MiB        │ 28.0MiB          │
└────────────┴────────────────┴──────────────────┘
```

### onnx >>> ncnn
create own project folder
```bash
mkdir my-ncnn && cd my-ncnn
cd my-ncnn
mkdir bin
mkdir models
```
copy `onnx2ncnn` converting tools into own directory
```bash
cp /home/ncnn/build/tools/onnx/onnx2ncnn /home/my-ncnn/bin
```
copy YOLOv5 onnx model to project folder
```bash
cp /home/yolov5/yolov5s.onnx /home/my-ncnn/models/
cp /home/yolov5/yolov5s-sim.onnx /home/my-ncnn/models/
```
converting model from `onnx` to ncnn's `param` & `bin`
```bash
cd my-ncnn
bin/onnx2ncnn models/yolov5s-sim.onnx models/yolov5s.param models/yolov5s.bin
```
find the blob name
```bash
cd /home/my-ncnn/models
grep -w -e 0=1 yolov5s.param
```

```
Permute          /model.24/Transpose      1 1 /model.24/Reshape_output_0 /model.24/Transpose_output_0 0=1
Permute          /model.24/Transpose_1    1 1 /model.24/Reshape_2_output_0 /model.24/Transpose_1_output_0 0=1
Permute          /model.24/Transpose_2    1 1 /model.24/Reshape_4_output_0 /model.24/Transpose_2_output_0 0=1
```
`/model.24/Transpose_output_0`, `/model.24/Transpose_1_output_0`, `/model.24/Transpose_2_output_0`
are the blob name

### YOLOv5 post-processing
create source code folder
```bash
cd my-ncnn
mkdir src && cd src
touch yolov5-ncnn.cpp
```

### Build own YOLOv5-ncnn
create `CMakeLists.txt`
```cmake

```
create `build` folder
```bash
cd my-ncnn
mkdir build && cd build
cmake ..
make -j16
```
create `images` folder   
place test image in this folder
```bash
mkdir images
```

## Error handling
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
