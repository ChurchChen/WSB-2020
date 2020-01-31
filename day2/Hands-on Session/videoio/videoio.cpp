#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	// 打开第一个摄像头
	VideoCapture cap(0);
	// 打开视频文件
	//VideoCapture cap(argv[1]);
	// 检查是否打开成功
	if(!cap.isOpened())
	{
		fprintf(stderr, "Failed to open camera or video file.\n");
		return -1;
	}

	// printf("frame count %d\n", int(cap.get(CAP_PROP_FRAME_COUNT)));

	VideoWriter writer;
	if(argc == 2)
	{
		Size s = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),
					  (int) cap.get(CAP_PROP_FRAME_HEIGHT));

		// 创建writer，并指定FOURCC、FPS、帧大小等参数
		writer = VideoWriter(argv[1], VideoWriter::fourcc('M','J','P','G'), 25, s);
		//检查是否创建成功
		if(!writer.isOpened())
		{
			 fprintf(stderr, "Failed to create video file.\n");
			 return -1;
		}
	}

	Mat frame, frame_proc;
    int cnt = 0;
	for(;;)
	{
		// 从cap读取一帧，存入frame
		cap >> frame;
		// 如果未读取到图像
		if(frame.empty())
			break;

        cnt++;

		// 进行图像处理
		// frame_proc = frame.clone();
		// 转为灰度图（单通道）
		// cvtColor(frame_proc, frame_proc, COLOR_BGR2GRAY);
		// 边缘提取
		// Canny(frame_proc, frame_proc, 0, 30, 3);
		//  也可进行其他图像分析，如人脸检测等

		// 显示图像
		imshow("camera", frame);

		// 等待1ms，如果'Esc'按下则退出循环
        switch (waitKey(1))
        {
        case 27:    // 按下Esc键，退出程序
            return 0;
        case 83:    // 按下S（大写）键，保存当前帧图像
            printf("saving frame\n");
            imwrite("frame_" + to_string(cnt) + ".jpg", frame);
            break;
        default:
            break;
        }

		if(argc == 2)
			// 将原始图像写入视频文件
			writer << frame;
			// 将处理后的图像写入视频文件
			// cvtColor(frame_proc, frame_proc, COLOR_GRAY2BGR);
			// writer << frame_proc;
	}

	// 退出时会自动释放cap和writer占用的资源
	return 0;
}