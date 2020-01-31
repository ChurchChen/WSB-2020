#include <string>
#include "opencv2/opencv.hpp"
#include "mtcnn.hpp"
#include "mtcnn_utils.hpp"
#include "utils.hpp"

#define EYE_POS_Y 0.3
#define LEFT_EYE_POS_X 0.3
#define RIGHT_EYE_POS_X 0.7
#define FACE_SIZE_X 96
#define FACE_SIZE_Y 96
#define FEAT_DIM 128

using namespace cv;
using namespace dnn;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] <<
            " <model_dir> <database_dir> <registration>" << std::endl;
        return 1;
    }
    std::string model_dir = argv[1];
    std::string database_dir = argv[2];

    int registration = 0;
    if (argc > 3)
        registration = std::stoi(argv[3]);

    // set mtcnn params
    int min_size = 40;

    float conf_p = 0.6;
    float conf_r = 0.7;
    float conf_o = 0.7;

    float nms_p = 0.5;
    float nms_r = 0.7;
    float nms_o = 0.7;
    
    mtcnn* det = new mtcnn(min_size, conf_p, conf_r, conf_o, nms_p, nms_r, nms_o);
    // load face&landmark detection model
    det->load_3model(model_dir);

    // load feature extractor model
    Net featEmbedder = readNet(model_dir + "openface/openface_nn4.small2.v1.t7");
    featEmbedder.setPreferableBackend(DNN_BACKEND_OPENCV);
    featEmbedder.setPreferableTarget(DNN_TARGET_CPU);

    // similarity threshold
    float sthre = 0.8;

    if (registration)
    {
    }

    // load facedb
    std::vector<Mat> facedbFt;
    std::vector<std::string> facedbNames;
    loadFacedb(facedbFt, facedbNames, database_dir, featEmbedder);

    float eyeCentres_ref[] = {LEFT_EYE_POS_X * FACE_SIZE_X,
                            EYE_POS_Y * FACE_SIZE_Y,
                            RIGHT_EYE_POS_X * FACE_SIZE_X,
                            EYE_POS_Y * FACE_SIZE_Y};

    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Fail to connect camera." << std::endl;
        return 1;
    }

    Mat im;
    while (waitKey(1) < 0)
    {
        cap >> im;
        if (im.empty())
        {
            std::cerr << "Fail to capture image." << std::endl;;
            break;
        }

        std::vector<face_box> face_info;
        // run mtcnn face and landmark detection
        det->detect(im, face_info);

        // draw detection results
        if (!face_info.empty())
        {
            for (unsigned int i = 0; i < face_info.size(); i++)
            {
                face_box& box = face_info[i];

                // face alignment
                // Mat face = im(Rect(Point(box.x0, box.y0), Point(box.x1, box.y1)));
                Mat faceAligned;
                float eyeCentres[] = { box.landmark.x[0], box.landmark.y[0],
                                    box.landmark.x[1], box.landmark.y[1] };
                faceAlignment(im, faceAligned, eyeCentres, eyeCentres_ref,
                            Size(FACE_SIZE_X, FACE_SIZE_Y));

                // feature extraction
                Mat blob_faceAligned;
                blobFromImage(faceAligned, blob_faceAligned, 1. / 255., Size(), Scalar(), true, false);
                featEmbedder.setInput(blob_faceAligned);
                Mat featA = featEmbedder.forward();

                float minD = 1.0;
                int faceIdx = -1;
                for (int idx = 0; idx < 5; idx++)
                {
                    Mat featB = facedbFt[idx];
                    double d = norm(featA - featB);
                    if (d < minD)
                    {
                        minD = d;
                        faceIdx = idx;
                    }
                }

                std::string name;
                if (minD > sthre)
                    name = "unknown";
                else
                    name = facedbNames[faceIdx];
                std::cout << name << " " << minD << std::endl;

                // draw results
                // face bounding box
                rectangle(im, Point(box.x0, box.y0), Point(box.x1, box.y1), Scalar(0, 255, 0), 2);
                // facial landmarks
                for (int l = 0; l < 5; l++)
                    circle(im, Point(box.landmark.x[l], box.landmark.y[l]), 2, Scalar(255, 255, 255), 2);
                // recognition
                putText(im, name, Point(box.x0, box.y0), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
            }
        }

        imshow("Face Recognition", im);
    }

    delete det;
    destroyAllWindows();

    return 0;
}