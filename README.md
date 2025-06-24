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

## YOLOv5
### Install pytorch
depends on your device
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas opencv-python-headless pyyaml tqdm matplotlib seaborn onnx onnxsim protobuf
```

### Download `.pt` weights
```bash
# checkout yolov5 v7.0 project
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout v7.0

# install requirements
pip install -r requirements.txt --user

# download yolov5s weight
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### Export torchscript
```bash
# export to torchscript, result saved to yolov5s.torchscript
python export.py --weights yolov5s.pt --include torchscript
```

### use pnnx convert
```bash
# download latest pnnx from https://github.com/pnnx/pnnx/releases
wget https://github.com/pnnx/pnnx/releases/download/20250530/pnnx-20250530-linux.zip
unzip pnnx-20250530-linux.zip

# convert torchscript to pnnx and ncnn, result saved to yolov5s.ncnn.param yolov5s.ncnn.bin
./pnnx-20250530-linux/pnnx yolov5s.torchscript inputshape=[1,3,640,640]
```

### copy `.param`, `.bin`
```bash
cd work_dir
mkdir models && cd models

cp /home/yolov5/yolov5s.ncnn.param /home/models/yolov5s.ncnn.param
cp /home/yolov5/yolov5s.ncnn.bin /home/models/yolov5s.ncnn.bin
```

### build own YOLOv5 project
create `CMakeLists.txt`
```cmake
project(yolov5)
# TODO change me to your ncnn installation path
set(ncnn_DIR "/home/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(yolov5 yolov5.cpp)
target_link_libraries(yolov5 ncnn ${OpenCV_LIBS})
```

create source code
```bash
cd my-ncnn
touch yolov5.cpp
```
```cpp
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    // load ncnn model
    ncnn::Net yolov5;

    // yolov5.opt.use_vulkan_compute = true;

    yolov5.load_param("../models/yolov5s.ncnn.param");
    yolov5.load_model("../models/yolov5s.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    // load image, resize and pad to 640x640
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);

    // the out blob would be a 2-dim tensor with w=85 h=25200
    //
    //        |cx|cy|bw|bh|box score(1)| per-class scores(80) |
    //        +--+--+--+--+------------+----------------------+
    //        |53|50|70|80|    0.11    |0.1 0.0 0.0 0.5 ......|
    //   all /|  |  |  |  |      .     |           .          |
    //  boxes |46|40|38|44|    0.95    |0.0 0.9 0.0 0.0 ......|
    // (25200)|  |  |  |  |      .     |           .          |
    //       \|  |  |  |  |      .     |           .          |
    //        +--+--+--+--+------------+----------------------+
    //

    // enumerate all boxes
    std::vector<Object> proposals;
    for (int i = 0; i < out.h; i++)
    {
        const float* ptr = out.row(i);

        const int num_class = 80;

        const float cx = ptr[0];
        const float cy = ptr[1];
        const float bw = ptr[2];
        const float bh = ptr[3];
        const float box_score = ptr[4];
        const float* class_scores = ptr + 5;

        // find class index with the biggest class score among all classes
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int j = 0; j < num_class; j++)
        {
            if (class_scores[j] > class_score)
            {
                class_score = class_scores[j];
                class_index = j;
            }
        }

        // combined score = box score * class score
        float confidence = box_score * class_score;

        // filter candidate boxes with combined score >= prob_threshold
        if (confidence < prob_threshold)
            continue;

        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;

        // collect candidates
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = class_index;
        obj.prob = confidence;

        proposals.push_back(obj);
    }

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    // the out blob would be a 3-dim tensor with w=dynamic h=dynamic c=255=85*3
    // we view it as [grid_w,grid_h,85,3] for 3 anchor ratio types

    //
    //            |<--   dynamic anchor grids     -->|
    //            |   larger image yields more grids |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-0   | box-h                            |
    //  +-----+   | box score(1)                     |
    //  |     |   +----------------                  |
    //  |     |   | per-class scores(80)             |
    //  +-----+\  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-1   | box-h                            |
    //  +-----+   | box score(1)                     |
    //  |     |   +----------------                  |
    //  +-----+   | per-class scores(80)             |
    //         \  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-2   | box-h                            |
    //  +--+      | box score(1)                     |
    //  |  |      +----------------                  |
    //  |  |      | per-class scores(80)             |
    //  +--+   \  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //

    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                // combined score = box score * class score
                // apply sigmoid first to get normed 0~1 value
                float confidence = sigmoid(box_score) * sigmoid(class_score);

                // filter candidate boxes with combined score >= prob_threshold
                if (confidence < prob_threshold)
                    continue;

                // yolov5/models/yolo.py Detect forward
                // y = x[i].sigmoid()
                // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                float cx = (dx * 2.f - 0.5f + j) * stride;
                float cy = (dy * 2.f - 0.5f + i) * stride;

                float bw = pow(dw * 2.f, 2) * anchor_w;
                float bh = pow(dh * 2.f, 2) * anchor_h;

                // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
                float x0 = cx - bw * 0.5f;
                float y0 = cy - bh * 0.5f;
                float x1 = cx + bw * 0.5f;
                float y1 = cy + bh * 0.5f;

                // collect candidates
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = confidence;

                objects.push_back(obj);
            }
        }
    }
}

static int detect_yolov5_dynamic(const cv::Mat& bgr, std::vector<Object>& objects)
{
    // load ncnn model
    ncnn::Net yolov5;

    // yolov5.opt.use_vulkan_compute = true;

    yolov5.load_param("yolov5s.ncnn.param");
    yolov5.load_model("yolov5s.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    // load image, resize and letterbox pad to multiple of max_stride
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    const int max_stride = 64;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    const int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("187", out0);
    ex.extract("202", out1);
    ex.extract("218", out2);

    // ex.extract("196", out0);
    // ex.extract("210", out1);
    // ex.extract("224", out2);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out0, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out1, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out2, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov5(m, objects);
    // detect_yolov5_dynamic(m, objects);

    draw_objects(m, objects);

    return 0;
}
```

## Build own YOLOv5
create `CMakeLists.txt`
```cmake
project(yolov5)
# TODO change me to your ncnn installation path
set(ncnn_DIR "/home/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(yolov5 yolov5.cpp)
target_link_libraries(yolov5 ncnn ${OpenCV_LIBS})
```
create `build` folder
```bash
cd my-ncnn
mkdir build && cd build
cmake ..
make -j16
```

### Inference
```bash
cd bin
./yolov5 ../test.jpg
```

---
# Build for arm64
### Prepare cross compile shell environments
```bash
source /path/to/environment-setup-aarch64-poky-linux

(e.g.)
source ~/Renesas/rcar-xos/v3.35.0/tools/toolchains/poky/environment-setup-aarch64-poky-linux
```
clone NCNN source code
```bash
cd my-ncnn-arm64
git clone https://github.com/Tencent/ncnn.git
```
```bash
cd ncnn
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      ..
make -j$(nproc)
```
### Compile
```bash
cd my-ncnn-arm64
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      ..

make -j$(nproc)
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
