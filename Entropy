#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <stack>
#define _MATH_DEFINES_DEFINED
using namespace std;
using namespace cv;
using namespace Eigen;
float Otsu(const Mat& image, int T)
{
	int nr = image.rows;
	int nl = image.cols;
	Mat hist = Mat::zeros(1, 256, CV_32F);
	float* hi = hist.ptr<float>(0);
	float h1 = 0;//类一的熵
	float h2 = 0;//类二的熵
	float pt = 0;//类一的概率
	float pg = 0;//总熵
	for (int i = 0; i < nr; i++)
	{
		const float* im = image.ptr<float>(i);
		for (int j = 0; j < nl; j++)
		{
			hi[int(im[j])] = hi[int(im[j])] + 1;
		}
	}
	//cout << hist << endl;
	for (int i = 0; i <=T; i++)
	{
		pt = pt + hi[i] / nr / nl;
	}
	for (int i = 0; i <= T; i++)
	{
		float ppp = log10f(hi[i] / nr / nl / pt);
		if(hi[i] / nr / nl / pt<0.00001)
		{ }
		else h1 = h1 - hi[i] / nr / nl / pt * log10f(hi[i] / nr / nl / pt);
	}
	for (int i = T + 1; i < 256; i++)
	{
		if (hi[i] / nr / nl / (1-pt) < 0.00001)
		{
		}
		else h2 = h2 - hi[i] / nr / nl / (1 - pt) * log10f(hi[i] / nr / nl / (1 - pt));
	}
	pg = h1 + h2;
	return pg;
}
int getT(const Mat& image, Mat& new_image)
{
	int temp = 0;
	float max = 0;
	max = Otsu(image, 40);
	for (int i = 40; i < 240; i++)//寻找最大的分割点
	{
		float gg = Otsu(image, i);
		if (max < gg)
		{
			temp = i;
			max = gg;
		}
	}
	for (int i = 0; i < image.rows; i++)//二值化
	{
		const float* im = image.ptr<float>(i);
		float* p = new_image.ptr<float>(i);
		for (int j = 0; j < image.cols; j++)
		{
			float ge = im[j];
			if (ge <= temp)
			{
				p[j] = 0;
			}
			else
			{
				p[j] = 255;
			}

		}
	}
	return temp;
}
int main()
{
	Mat image = imread("I:/F.jpg", IMREAD_GRAYSCALE);
	imshow("原图像", image);
	Mat IM;
	image.convertTo(IM, CV_32F);
	//imshow("fsSSd", IM);
	Mat new_image = Mat::zeros(IM.rows, IM.cols, CV_32F);
	int T = getT(IM, new_image);
	cout << T << endl;
	imshow("新图像", new_image);
	cv::waitKey(0);
}
