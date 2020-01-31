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

    // ������������ṩ������������Ƶ��ļ���
    // �����ƴ���classes��
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
    

    // ��ȡ����ģ��
    Net net = readNet(modelPath, configPath);
    // ѡ��ģ�ͼ���ĺ��
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    // ѡ��ģ�ͼ�����豸
    net.setPreferableTarget(DNN_TARGET_CPU);    

    // ��������
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    int initialConf = (int)(confThreshold * 100);
    // �����϶���
    createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, &confThreshold);

    VideoCapture cap(0);

    // ��ȡ����ģ�����������
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        // ����ͼ��
        cap >> frame;
        // ��δ��ȡ��ͼ����ȴ�������Ӧ��Ȼ���˳�����
        if (frame.empty())
        {
            waitKey();
            break;
        }

        Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
                     inpHeight > 0 ? inpHeight : frame.rows);
        // ������ͼ������4D blob ��4ά���󣬰�NCHW��ͼ��������ͨ������ͼ��߶ȣ�ͼ���ȣ�
        blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
        // ����ģ�͵�����
        net.setInput(blob);
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
        {
            resize(frame, frame, inpSize);
            Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        // ����ģ��ǰ����㣨����
        // �õ�outNames��ָ������������outs��
        std::vector<Mat> outs;
        net.forward(outs, outNames);

        // ��ģ�͵�������д���ȥ�����Ŷȵ÷ֵ͵ĺ�ĳЩ�ص��ȸߵľ��ο�
        postprocess(frame, outs, net, confThreshold, nmsThreshold, classes);

        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        // ��ȡ����ʱ��
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        // ������ʱ����ʾ��ͼ����
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
            // ��������Ϊ1x1xNx7��blob
            // NΪ��������
            // ÿ�������7��ֵ��ɣ�[batchId, classId, confidence, left, top, right, bottom]
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
                    classIds.push_back((int)(data[i + 1]) - 1);  // ������0����𣨱���������
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
            // ����ģ�����ΪNxC��blob
            // N�Ǽ��������������
            // C=�����+4��ǰ4��ֵΪ[center_x, center_y, width, height]��ʣ�µ�ֵ��Ӧ
            // ÿ���������Ŷȵ÷�
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // �ҵ�����Ԫ�ص����ֵ����λ��
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
    // �Ǽ���ֵ����
    // nmsThreshold�������ȥ�����ο�������õ�̫�ͣ��򽫻ᱣ�������ص��ľ��ο�
    // ������õ�̫�ߣ������׽��ٽ�����ľ��ο�ȥ������
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        // ��ͼ���ϻ��Ƽ�������������ο�����������Ŷ�
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classes);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, std::vector<std::string>& classes)
{
    // ��������ľ��ο�
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
    // ������ʾ�����������Ϣ�ľ��ο�
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    // ��д�������������Ϣ
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

// �ص��������������Ŷ���ֵ
void callback(int pos, void* thre)
{
    *((float*)thre) = pos * 0.01f;
}
