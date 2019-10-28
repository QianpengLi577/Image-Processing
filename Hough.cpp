//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <Eigen/Dense>
//#include <math.h>
//#include <stack>
//#define _MATH_DEFINES_DEFINED
//using namespace std;
//using namespace cv;
//using namespace Eigen;
//void countpt(const Mat& image, Mat& pt)
//{
//	int nr = image.rows;
//	int nl = image.cols;
//	int ml = int( sqrt(nr * nr + nl * nl));
//	for (int i = 20; i < nr-20; i++)
//	{
//		const float* im = image.ptr<float>(i);
//		for (int j = 20; j < nl-20; j++)
//		{
//			if(im[j]>50)
//			{
//				for (int x = 0; x < 180; x++)
//				{
//					float* pt_im = pt.ptr<float>(x);
//					int py = int(i * cos(x * M_PI / 180) + j * sin(x * M_PI / 180) + ml);
//					pt_im[py] = pt_im[py] + 1;
//				}
//			}
//			
//		}
//	}
//}
//
//int countmax(const Mat& pt)
//{
//	int nr = pt.rows;
//	int nl = pt.cols;
//	int max = 0;
//	for (int i = 0; i < nr; i++)
//	{
//		const float* ppt = pt.ptr<float>(i);
//		for (int j = 0; j < pt.cols; j++)
//		{
//			if (max < ppt[j])
//			{
//				max = ppt[j];
//			}
//		}
//	}
//	return max;
//}
//
//void findmax(Mat& pt, stack<int >& pp,int max)
//{
//	int nr = pt.rows;
//	int nl = pt.cols;
//	for (int i = 0; i < nr; i++)
//	{
//		const float* ppt = pt.ptr<float>(i);
//		for (int j = 0; j < pt.cols; j++)
//		{
//			//更改判断阈值，可以达到输出多少条直线的效果
//			if (0.65*max < ppt[j]){
//				pp.push(i);
//				pp.push(j);
//			}
//		}
//	}
//}
//
//bool findok(int r, int l, int T, const Mat& image)
//{
//	int temp = 0;
//	for (int ii = r - T; ii < r + T; ii++)
//	{
//		if (ii > 0 && ii < image.rows)
//		{
//			const float* p = image.ptr<float>(ii);
//			for (int jj = l - T; jj < l + T; jj++)
//			{
//				if ((p[jj] > 50) && jj > 0 && jj < image.cols)
//				{
//					temp = temp + 1;
//				}
//			}
//		}
//	}
//	if (temp == 0)return false;
//	if(temp>2*T*T) return true;
//}
//
//void hough(Mat& new_image, stack<int >&pp,const Mat&image)
//{
//	int nr = new_image.rows;
//	int nl = new_image.cols;
//	int esplion = 1;
//	while (!pp.empty()) {
//		int y = pp.top()-int(sqrt(image.rows * image.rows + image.cols * image.cols));
//		pp.pop();
//		int x = pp.top();
//		pp.pop();
//		for (int i = 0; i < nr; i++)
//		{
//			float* im = new_image.ptr<float>(i);
//			for (int j = 0; j < nl; j++)
//			{
//				if ((abs(i * cos(x * M_PI / 180) + j * sin(x * M_PI / 180) - y) < esplion) && findok(i, j, 1, image))
//				{
//					im[j] = 255;
//				}
//			}
//		}
//	}
//}
//
//int main()
//{
//	Mat Image = imread("I://Q.png", IMREAD_GRAYSCALE);
//	Mat image;
//	Image.convertTo(image, CV_32FC1);
//	imshow("原图", image);
//	Mat H = Mat::zeros(180, int(2*sqrt(image.rows*image.rows+image.cols*image.cols)), CV_32FC1);
//	countpt(image, H);
//	int max=countmax(H);
//	stack<int >CT;
//	findmax(H, CT, max);
//	Mat new_image = Mat::zeros(image.rows, image.cols, CV_32F);
//	hough(new_image, CT,image);
//	new_image.convertTo(new_image, CV_8U);
//	imshow("新图", new_image);
//	imwrite("hough.jpg", new_image);
//	
//	waitKey(0);
//	cout << "结束" << endl;
//}
