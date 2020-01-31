/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

/* modified for using OpenCV DNN
   jia.wu@opencv.org.cn */

#include "mtcnn.hpp"
#include "mtcnn_utils.hpp"

using namespace cv;
using namespace dnn;

mtcnn::mtcnn(int minsize, float conf_p, float conf_r, float conf_o, float nms_p, float nms_r, float nms_o)
{
    minsize_ = minsize;

    conf_p_threshold_ = conf_p;
    conf_r_threshold_ = conf_r;
    conf_o_threshold_ = conf_o;

    nms_p_threshold_ = nms_p;
    nms_r_threshold_ = nms_r;
    nms_o_threshold_ = nms_o;
}

int mtcnn::load_3model(const std::string& model_dir)
{
    std::string proto_name, mdl_name;

    // load P-Net
    proto_name = model_dir + "mtcnn/det1.prototxt";
    mdl_name = model_dir + "mtcnn/det1.caffemodel";
    PNet = readNet(mdl_name, proto_name);
    PNet.setPreferableBackend(DNN_BACKEND_OPENCV);
    PNet.setPreferableTarget(DNN_TARGET_CPU);

    // load R-Net
    proto_name = model_dir + "mtcnn/det2.prototxt";
    mdl_name = model_dir + "mtcnn/det2.caffemodel";
    RNet = readNet(mdl_name, proto_name);
    RNet.setPreferableBackend(DNN_BACKEND_OPENCV);
    RNet.setPreferableTarget(DNN_TARGET_CPU);

    // load O-Net
    proto_name = model_dir + "mtcnn/det3.prototxt";
    mdl_name = model_dir + "mtcnn/det3.caffemodel";
    ONet = readNet(mdl_name, proto_name);
    ONet.setPreferableBackend(DNN_BACKEND_OPENCV);
    ONet.setPreferableTarget(DNN_TARGET_CPU);

    return 0;
}

int mtcnn::run_PNet(const Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
    int scale_h = win.h;
    int scale_w = win.w;
    float scale = win.scale;

    Mat imgBlob;
    blobFromImage(img, imgBlob, 1.0, Size(scale_w, scale_h), Scalar(), false, false);
    PNet.setInput(imgBlob);

    // run model and get the outputs
    std::vector<std::string> outNames;
    outNames.push_back("conv4-2");
    outNames.push_back("prob1");
    std::vector<Mat> outs;
    PNet.forward(outs, outNames);

    Mat coords = outs[0];   // box regression
    Mat probs = outs[1];    // scores

    std::vector<face_box> candidate_boxes;
    generate_boudning_box(coords, probs, scale, conf_p_threshold_, candidate_boxes);
    nms_boxes(candidate_boxes, nms_p_threshold_, NMS_UNION, box_list);
    // NMSBoxes();

    return 0;
}

int mtcnn::run_RNet(const Mat& img, std::vector<face_box>& pnet_boxes,
                    std::vector<face_box>& output_boxes)
{
    int batchSize = pnet_boxes.size();
    int height = 24;
    int width = 24;

    // crop image to get proposal faces
    std::vector<Mat> proposals;
    for (unsigned int i = 0; i < pnet_boxes.size(); i++)
    {
        face_box roi = pnet_boxes[i];
        roi.x0 = roi.x0 < 0 ? 0 : roi.x0;
        roi.y0 = roi.y0 < 0 ? 0 : roi.y0;
        roi.x1 = roi.x1 > (img.cols - 1) ? img.cols - 1 : roi.x1;
        roi.y1 = roi.y1 > (img.rows - 1) ? img.rows - 1 : roi.y1;
        Mat faceImg;
        img(Rect(Point2i(roi.px0, roi.py0), Point2i(roi.px1, roi.py1))).copyTo(faceImg);

        int pad_top = std::abs(roi.py0 - roi.y0);
        int pad_left = std::abs(roi.px0 - roi.x0);
        int pad_bottom = std::abs(roi.py1 - roi.y1);
        int pad_right = std::abs(roi.px1 - roi.x1);

        copyMakeBorder(faceImg, faceImg, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT);
        resize(faceImg, faceImg, Size(width, height));

        proposals.push_back(faceImg);
    }

    Mat blob;
    blobFromImages(proposals, blob, 1.0, Size(), Scalar(), false, false);
    RNet.setInput(blob);

    // run model and get the outputs
    std::vector<std::string> outNames;
    outNames.push_back("conv5-2");
    outNames.push_back("prob1");
    std::vector<Mat> outs;
    RNet.forward(outs, outNames);

    Mat coords = outs[0];
    Mat probs = outs[1];

    // get results
    for (unsigned int i = 0; i < proposals.size(); i++)
    {
        float* pcoords = coords.ptr<float>(i);
        float prob = probs.ptr<float>(i)[1];
        if (prob > conf_r_threshold_)
        {
            face_box input_box = pnet_boxes[i];
            face_box output_box;

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.score = prob;

            output_box.regress[0] = pcoords[0];
            output_box.regress[1] = pcoords[1];
            output_box.regress[2] = pcoords[2];
            output_box.regress[3] = pcoords[3];

            output_boxes.push_back(output_box);
        }
    }
    
    return 0;
}

int mtcnn::run_ONet(const Mat& img, std::vector<face_box>& rnet_boxes,
                    std::vector<face_box>& output_boxes)
{
    int batchSize = rnet_boxes.size();
    int height = 48;
    int width = 48;

    // crop image to get proposal faces
    std::vector<Mat> proposals;
    for (unsigned int i = 0; i < rnet_boxes.size(); i++)
    {
        face_box roi = rnet_boxes[i];
        Mat faceImg;
        img(Rect(Point2i(roi.px0, roi.py0), Point2i(roi.px1, roi.py1))).copyTo(faceImg);
        roi.x0 = roi.x0 < 0 ? 0 : roi.x0;
        roi.y0 = roi.y0 < 0 ? 0 : roi.y0;
        roi.x1 = roi.x1 > (img.cols - 1) ? img.cols - 1 : roi.x1;
        roi.y1 = roi.y1 > (img.rows - 1) ? img.rows - 1 : roi.y1;

        int pad_top = std::abs(roi.py0 - roi.y0);
        int pad_left = std::abs(roi.px0 - roi.x0);
        int pad_bottom = std::abs(roi.py1 - roi.y1);
        int pad_right = std::abs(roi.px1 - roi.x1);

        copyMakeBorder(faceImg, faceImg, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT);
        resize(faceImg, faceImg, Size(width, height));

        proposals.push_back(faceImg);
    }

    Mat blob;
    blobFromImages(proposals, blob, 1.0, Size(), Scalar(), false, false);
    ONet.setInput(blob);

    // run model and get the outputs
    std::vector<std::string> outNames;
    outNames.push_back("conv6-2");
    outNames.push_back("conv6-3");
    outNames.push_back("prob1");
    std::vector<Mat> outs;
    ONet.forward(outs, outNames);

    Mat coords = outs[0];
    Mat landmarks = outs[1];
    Mat probs = outs[2];

    // get results
    for (unsigned int i = 0; i < proposals.size(); i++)
    {
        float* pcoords = coords.ptr<float>(i);
        float* plandmarks = landmarks.ptr<float>(i);
        float prob = probs.ptr<float>(i)[1];
        if (prob > conf_o_threshold_)
        {
            face_box input_box = rnet_boxes[i];
            face_box output_box;

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.score = prob;

            output_box.regress[0] = pcoords[0];
            output_box.regress[1] = pcoords[1];
            output_box.regress[2] = pcoords[2];
            output_box.regress[3] = pcoords[3];

            for (unsigned int j = 0; j < 5; j++)
            {
                output_box.landmark.x[j] = plandmarks[j + 5];
                output_box.landmark.y[j] = plandmarks[j];
            }

            output_boxes.push_back(output_box);
        }
    }

    return 0;
}

void mtcnn::detect(Mat img, std::vector<face_box>& face_list)
{
    Mat workImg;
    img.copyTo(workImg);
    cvtColor(workImg, workImg, COLOR_BGR2RGB);
    workImg.convertTo(workImg, CV_32FC3);

    float mean = 127.5;
    float deno = 0.0078125;
    workImg = (workImg - mean) * deno;
    workImg = workImg.t();

    // compute image pyramids
    std::vector<scale_window> win_list;
    cal_scale_list(workImg.rows, workImg.cols, minsize_, win_list);

    // run PNet on the image pyramids
    std::vector<face_box> total_pnet_boxes;;
    for(unsigned int i = 0; i < win_list.size(); i++)
    {
        std::vector<face_box> boxes;
        if(run_PNet(workImg, win_list[i], boxes) != 0)
            return;
        total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
    }
    win_list.clear();
    std::vector<face_box> pnet_boxes;
    process_boxes(total_pnet_boxes, workImg.rows, workImg.cols, pnet_boxes, 0.7f);

    if(!pnet_boxes.size())
        return;

    // run RNet on proposals from PNet
    std::vector<face_box> total_rnet_boxes;
    if(run_RNet(workImg, pnet_boxes, total_rnet_boxes) != 0)
         return;
    total_pnet_boxes.clear();

    std::vector<face_box> rnet_boxes;
    process_boxes(total_rnet_boxes, workImg.rows, workImg.cols, rnet_boxes, nms_r_threshold_);

    if(!rnet_boxes.size())
         return;

    // run ONet
    std::vector<face_box> total_onet_boxes;
    if(run_ONet(workImg, rnet_boxes, total_onet_boxes) != 0)
         return;
    total_rnet_boxes.clear();

    for(unsigned int i = 0; i < total_onet_boxes.size(); i++)
    {
         face_box& box = total_onet_boxes[i];

         float w = box.x1 - box.x0 + 1.f;
         float h = box.y1 - box.y0 + 1.f;

         for(int j = 0; j < 5; j++)
         {
             box.landmark.x[j] = box.x0 + w * box.landmark.x[j] - 1;
             box.landmark.y[j] = box.y0 + h * box.landmark.y[j] - 1;
         }
     }
    
    regress_boxes(total_onet_boxes);
    nms_boxes(total_onet_boxes, nms_o_threshold_, NMS_MIN, face_list);
    square_boxes(face_list);
    total_onet_boxes.clear();

    for(unsigned int i = 0; i < face_list.size(); i++)
    {
         face_box& box = face_list[i];

         std::swap(box.x0, box.y0);
         std::swap(box.x1, box.y1);

         for(int l = 0; l < 5; l++)
         {
             std::swap(box.landmark.x[l], box.landmark.y[l]);
         }
     }
}
