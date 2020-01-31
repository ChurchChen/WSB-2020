#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "opencv2/opencv.hpp"

#define PI 3.14159265

using namespace cv;


void loadFacedb(std::vector<Mat>& facedbFt, std::vector<std::string>& facedbNames,
                std::string databaseDir, dnn::Net& ftModel)
{
    std::string db_faces[] = {databaseDir + "chenhuishan.jpg",
                            databaseDir + "gutianle.jpg",
                            databaseDir + "wujia.jpg",
                            databaseDir + "yuanyongyi.jpg",
                            databaseDir + "zhangxueyou.jpg" };

    facedbNames.push_back("chenhuishan");
    facedbNames.push_back("gutianle");
    facedbNames.push_back("wujia");
    facedbNames.push_back("yuanyongyi");
    facedbNames.push_back("zhangxueyou");

    for (int i = 0; i < 5; i++)
    {
        Mat facedbImg = imread(db_faces[i]);
        Mat facedbBlob;
        dnn::blobFromImage(facedbImg, facedbBlob, 1. / 255., Size(), Scalar(), true, false);
        ftModel.setInput(facedbBlob);
        Mat ft;
        (ftModel.forward()).copyTo(ft);
        facedbFt.push_back(ft);
    }  
}

void faceAlignment(const Mat& img, Mat& faceImgAligned, 
                    float* eyeCenters, float* eyeCenters_ref, 
                    Size faceSize)
{
    float dist_ref = eyeCenters_ref[2] - eyeCenters_ref[0];
    float dx = eyeCenters[2] - eyeCenters[0];
    float dy = eyeCenters[3] - eyeCenters[1];
    float dist = sqrt(dx * dx + dy * dy);

    // scale
    double scale = dist_ref / dist;
    // angle
    double angle = atan2(dy, dx) * 180 / PI;
    // center
    Point2f center = Point2f(0.5 * (eyeCenters[0] + eyeCenters[2]),
                            0.5 * (eyeCenters[1] + eyeCenters[3]));
    // calculate rotation matrix
    Mat rot = getRotationMatrix2D(center, angle, scale);
    // translation
    rot.at<double>(0, 2) += faceSize.width * 0.5 - center.x;
    rot.at<double>(1, 2) += eyeCenters_ref[1] - center.y;    

    // apply affine transform
    cv::Mat imgIn = img.clone();
    imgIn.convertTo(imgIn, CV_32FC3, 1. / 255.);
    warpAffine(imgIn, faceImgAligned, rot, faceSize);
    faceImgAligned.convertTo(faceImgAligned, CV_8UC3, 255);
}

#endif // UTILS_HPP_