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

/* modified
 * jia.wu@opencv.org.cn
 */

#include "mtcnn_utils.hpp"

using namespace cv;

void cal_scale_list(int height, int width, int minsize, std::vector<scale_window>& list)
{
    float factor = 0.709;
    int MIN_DET_SIZE = 12;
    int minl = height < width ? height : width;
    float m = ( float )MIN_DET_SIZE / minsize;
    minl = minl * m;
    int factor_count = 0;
    while(minl > MIN_DET_SIZE)
    {
        if(factor_count > 0)
            m = m * factor;
        minl *= factor;
        factor_count++;

        scale_window win;
        win.h = ( int )ceil(height * m);
        win.w = ( int )ceil(width * m);
        win.scale = m;
        list.push_back(win);
    }
}

void generate_boudning_box(const Mat& coords, const Mat& probs, 
    float scale, float thre, std::vector<face_box>& output)
{
    int faceSize = 12;
    int stride = 2;

    int featmap_h = coords.size[2];
    int featmap_w = coords.size[3];
    int featmap_size = featmap_h * featmap_w;

    const float* pcoords = coords.ptr<float>(0);
    const float* pprobs = probs.ptr<float>(0, 1);
    for (int y = 0; y < featmap_h; y++)
    {
        for (int x = 0; x < featmap_w; x++)
        {
            int idx = y * featmap_w + x;

            float prob = pprobs[idx];
            if (prob > thre)
            {
                float tmp_x = x * stride;
                float tmp_y = y * stride;
                float top_x = tmp_x / scale;
                float top_y = tmp_y / scale;
                float bottom_x = (tmp_x + faceSize - 1.0f) / scale;
                float bottom_y = (tmp_y + faceSize - 1.0f) / scale;

                face_box box;
                box.x0 = top_x;
                box.y0 = top_y;
                box.x1 = bottom_x;
                box.y1 = bottom_y;
                box.score = pprobs[idx];
                box.regress[0] = pcoords[idx];
                box.regress[1] = pcoords[featmap_size + idx];
                box.regress[2] = pcoords[2 * featmap_size + idx];
                box.regress[3] = pcoords[3 * featmap_size + idx];

                output.push_back(box);
            }
        }
    }
}

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>& output)
{
    output.clear();
    std::sort(input.begin(), input.end(), [](const face_box& a, const face_box& b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for(int i = 0; i < box_num; i++)
    {
        if(merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y1 - input[i].y0 + 1;
        float w0 = input[i].x1 - input[i].x0 + 1;

        float area0 = h0 * w0;

        for(int j = i + 1; j < box_num; j++)
        {
            if(merged[j])
                continue;

            float inner_x0 = std::max(input[i].x0, input[j].x0);
            float inner_y0 = std::max(input[i].y0, input[j].y0);

            float inner_x1 = std::min(input[i].x1, input[j].x1);
            float inner_y1 = std::min(input[i].y1, input[j].y1);

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if(inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y1 - input[j].y0 + 1;
            float w1 = input[j].x1 - input[j].x0 + 1;

            float area1 = h1 * w1;

            float score;

            if(type == NMS_UNION)
            {
                score = inner_area / (area0 + area1 - inner_area);
            }
            else
            {
                score = inner_area / std::min(area0, area1);
            }

            if(score > threshold)
                merged[j] = 1;
        }
    }
}

void regress_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        face_box& box = rects[i];

        float h = box.y1 - box.y0 + 1.;
        float w = box.x1 - box.x0 + 1.;

        // !!!
        box.x0 = box.x0 + w * box.regress[1];
        box.y0 = box.y0 + h * box.regress[0];
        box.x1 = box.x1 + w * box.regress[3];
        box.y1 = box.y1 + h * box.regress[2];
    }
}

void square_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        float h = rects[i].y1 - rects[i].y0 + 1;
        float w = rects[i].x1 - rects[i].x0 + 1;

        float l = std::max(h, w);

        rects[i].x0 = rects[i].x0 + (w - l) * 0.5;
        rects[i].y0 = rects[i].y0 + (h - l) * 0.5;
        rects[i].x1 = rects[i].x0 + l - 1;
        rects[i].y1 = rects[i].y0 + l - 1;
    }
}

void padding(int img_h, int img_w, std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        rects[i].px0 = std::max(rects[i].x0, 1.0f);
        rects[i].py0 = std::max(rects[i].y0, 1.0f);
        rects[i].px1 = std::min(rects[i].x1, ( float )img_w);
        rects[i].py1 = std::min(rects[i].y1, ( float )img_h);
    }
}

void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects, float nms_r_thresh)
{
    nms_boxes(input, nms_r_thresh, NMS_UNION, rects);

    regress_boxes(rects);

    square_boxes(rects);

    padding(img_h, img_w, rects);
}
