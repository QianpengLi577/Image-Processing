#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <stack>
#define _MATH_DEFINES_DEFINED
using namespace std;
using namespace cv;
using namespace Eigen;
float Otsu(const Mat& image,int T)
{
	int nr = image.rows;
	int nl = image.cols;
	Mat hist = Mat::zeros(1, 256, CV_32F);
	float* hi = hist.ptr<float>(0);
	float junzhi=0;//定义总均值，类一均值，类二均值
	float junzhi0 = 0;
	float junzhi1 = 0;
	float lei0 = 0;//定义类一，类二的概率
	float lei1 = 0;
	float fangcha0=0;//定义类一，类二的方差
	float fangcha1 = 0;
	float neifangcha = 0;//定义类内方差
	float jianfangcha = 0;//定义类间方差
	float zongfangcha = 0;//定义总方差，类内加类间
	float ppppp = 0;
	for (int i = 0; i < nr; i++)//这里是统计直方图
	{
		const float* im = image.ptr<float>(i);
		for (int j = 0; j < nl; j++)
		{
			hi[int(im[j])] = hi[int(im[j])] +1;
		}
	}
	//cout << hist << endl;
	for (int i = 0; i < 256; i++)//计算均值
	{
		junzhi = i * hi[i] / nr / nl;
	}
	for (int i = 0; i <= T; i++)//计算
	{
		lei0 = lei0 + hi[i] / nr / nl;
	}
	for (int i = T+1; i <256; i++)
	{
		lei1 = lei1 + hi[i] / nr / nl;
	}
	for (int i = 0; i <= T; i++)
	{
		junzhi0 = junzhi0 + i * hi[i] / nr / nl / lei0;
	}
	for (int i = T + 1; i < 256; i++)
	{
		junzhi1 = junzhi1 + i * hi[i] / nr / nl / lei1;
	}
	for (int i = 0; i <= T; i++)
	{
		fangcha0 = fangcha0 + (i -junzhi0)* (i - junzhi0) * hi[i] / nr / nl / lei0;
	}
	for (int i = T + 1; i < 256; i++)
	{
		fangcha1 = fangcha1 + (i - junzhi1) * (i - junzhi1) * hi[i] / nr / nl / lei1;
	}
	neifangcha = lei0 * fangcha0 + lei1 * fangcha1;
	jianfangcha = lei0 * lei1 * (junzhi1 - junzhi0) * (junzhi1 - junzhi0);
	zongfangcha = neifangcha + jianfangcha;
	ppppp= jianfangcha / zongfangcha;
	return ppppp;
}
int getT(const Mat& image, Mat& new_image)
{
	int temp = 0;
	float max = 0;
	max = Otsu(image, 20);
	for (int i =  0; i < 255; i++)
	{
		float gg = Otsu(image, i);
		if (max < gg)
		{
			temp = i;
			max =gg;
		}
	}//遍历得到最大的分割点temp
	for (int i = 0; i < image.rows; i++)
	{
		const float* im =image.ptr<float>(i);
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
	}//图像进行二值化
	return temp;
}
int main()
{
	Mat image = imread("I:/C.jpg", IMREAD_GRAYSCALE);
	imshow("原图", image);
	Mat IM;
	image.convertTo(IM, CV_32F);
	//imshow("fsSSd", IM);
	Mat new_image = Mat::zeros(IM.rows, IM.cols, CV_32F);
	int T = getT(IM, new_image);
	cout << T << endl;
	imshow("分割以后", new_image);
	cv::waitKey(0);
}
