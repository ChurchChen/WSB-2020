#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "common.hpp"


using namespace cv;
using namespace dnn;


void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, 
                 float confThreshold, float nmsThreshold, std::vector<std::string>& classes);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame,
              std::vector<std::string>& classes);
void callback(int pos, void* thre);

int main(int argc, char** argv)
{
    std::string modelPath = "../ssdlite_mobilenet_v2/frozen_inference_graph.pb";
    std::string configPath = "../ssdlite_mobilenet_v2/graph.pbtxt";

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    float scale = 1.0;
    Scalar mean = Scalar(122.6778, 116.6522, 103.9997);
    bool swapRB = true;
    int inpWidth = 300;
    int inpHeight = 300;

    // 如果输入命令提供了物体类别名称的文件，
    // 则将名称存入classes中
    std::vector<std::string> classes;
    std::string file = "../ssdlite_mobilenet_v2/object_detection_classes_coco.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    

    // 读取网络模型
    Net net = readNet(modelPath, configPath);
    // 选择模型计算的后端
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    // 选择模型计算的设备
    net.setPreferableTarget(DNN_TARGET_CPU);    

    // 创建窗口
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    int initialConf = (int)(confThreshold * 100);
    // 创建拖动条
    createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, &confThreshold);

    VideoCapture cap(0);

    // 获取网络模型输出的名称
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        // 读入图像
        cap >> frame;
        // 若未获取到图像则等待键盘响应，然后退出程序
        if (frame.empty())
        {
            waitKey();
            break;
        }

        Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
                     inpHeight > 0 ? inpHeight : frame.rows);
        // 从输入图像生成4D blob （4维矩阵，按NCHW：图像数量，通道数，图像高度，图像宽度）
        blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
        // 设置模型的输入
        net.setInput(blob);
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
        {
            resize(frame, frame, inpSize);
            Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        // 进行模型前向计算（推理）
        // 得到outNames中指定层的输出放入outs中
        std::vector<Mat> outs;
        net.forward(outs, outNames);

        // 对模型的输出进行处理，去除置信度得分低的和某些重叠度高的矩形框
        postprocess(frame, outs, net, confThreshold, nmsThreshold, classes);

        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        // 获取推理时间
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        // 将计算时间显示在图像上
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, 
                 float confThreshold, float nmsThreshold, std::vector<std::string>& classes)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            // 网络的输出为1x1xNx7的blob
            // N为检测的数量
            // 每个检测由7个值组成：[batchId, classId, confidence, left, top, right, bottom]
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width * height <= 1)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // 跳过第0个类别（背景）索引
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        CV_Assert(outs.size() > 0);
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // 网络模型输出为NxC的blob
            // N是检测出的物体的数量
            // C=类别数+4，前4个值为[center_x, center_y, width, height]，剩下的值对应
            // 每个类别的置信度得分
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // 找到所有元素的最大值及其位置
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    // 非极大值抑制
    // nmsThreshold控制如何去除矩形框。如果设置得太低，则将会保留大量重叠的矩形框；
    // 如果设置得太高，则容易将临近物体的矩形框去除掉。
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        // 在图像上绘制检测结果，包括矩形框、物体类别、置信度
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classes);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, std::vector<std::string>& classes)
{
    // 绘制物体的矩形框
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    // 绘制显示检测结果文字信息的矩形框
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    // 书写检测结果的文字信息
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

// 回调函数，调节置信度阈值
void callback(int pos, void* thre)
{
    *((float*)thre) = pos * 0.01f;
}
