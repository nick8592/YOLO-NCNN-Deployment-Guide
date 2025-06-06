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
