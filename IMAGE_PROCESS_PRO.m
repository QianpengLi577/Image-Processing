%{
 *Copyright: Copyright (c) 2019  All rights reserved
 *Created on 2019-9-28  
 *Author:LiQianpeng<Qianpengli577@gmail.com>
 *Version 1.0 
%}
clc,clear;
I=imread('I:\A.jpg');%读取图像的信息

A1=HF(I(:,:,1));
A2=HF(I(:,:,2));
A3=HF(I(:,:,3));%提取三通道图像并进行同态滤波

picture=rgb2gray(I);%将图像信息进行灰度化
hist=count_hist(picture);%计算直方图
figure('Name','直方图')
stem(hist(2,:),hist(1,:),'.');%打印直方图
ylim([0 2*10^4])%限制直方图的Y值的上界限

[new_picture,new_hist]=hist_equal(picture,hist);%均衡化，返回新的图象以及直方图
figure('Name','均衡化后的直方图')
stem(new_hist(2,:),new_hist(1,:),'.');%打印均衡化之后的直方图
ylim([0 2*10^4])

figure('Name','均衡化后的图像')
imshow(new_picture)

FFT_picture=FFT(picture); %傅里叶变换，返回中心变换后的频率图
figure('Name','中心化频谱图')
imshow(log(abs(FFT_picture)+1),[])

IFFT_picture=IFFT(FFT_picture);
figure('Name','傅里叶逆变换图像')
imshow(uint8(abs(IFFT_picture)))

figure('Name','原图像')
imshow(picture)

HF_picture=HF(picture);
figure('Name','同态滤波后灰度图像')
imshow(HF_picture)

picture_new=zeros(size(A1,1),size(A1,2),3);  %矩阵拼接
picture_new(:,:,1)=A1;
picture_new(:,:,2)=A2;
picture_new(:,:,3)=A3;
figure('Name','同态滤波后彩色图像')
imshow(uint8(abs(picture_new)))

J=Region_Growing(I);
figure('Name','区域生长')
imshow(J);

JP=picture;
for i=1:size(picture,1)    %这是对区域分割之后，空域增强
    for j=1:size(picture,2)
        if J(i,j)==0   %0代表的是物体，对物体增加灰度值
            JP(i,j)=uint8(picture(i,j)+60);
        end
        if J(i,j)==1   %1代表的是背景， 对物体减少灰度值（10），作用应该不是很大
            JP(i,j)=uint8(picture(i,j)-10);
        end
    end
end
figure('Name','最终')
imshow(JP)  %这是改进算法的空域增强图像

JJ=HF(JP);   
figure('Name','最终在进行同态滤波')
imshow(uint8(JJ))

se=strel('square',3);%3*3的正方形
fo=imopen(J,se);%开运算
fc=imclose(J,se);%闭运算
fco=imopen(fc,se);%先闭后开
foc=imclose(fo,se);%先开后闭
figure('Name','fo')%开  图像
imshow(fo)
figure('Name','fc')%闭  图像
imshow(fc)
figure('Name','foc')%先开后闭  图像
imshow(foc)
figure('Name','fco')%先闭后开 图像
imshow(fco)

function hist=count_hist(picture) %计算直方图，输入变量为灰度值，返回矩阵为直方图
    Sum=zeros(2,256);%第一行表示的是出现的次数，第二行表示的是0~255供给256个灰度值
    for i=1:256
        Sum(2,i)=i-1;
    end   %第二行的初始化过程
    for i=1:size(picture,1)
        for j=1:size(picture,2)
            Sum(1,picture(i,j)+1)=Sum(1,picture(i,j)+1)+1;%由于灰度图像值是0~255，但索引为1~256，所以索引的时候需要编程X(i，j)+1
        end
    end  %对第一行进行赋值
    hist=Sum;
end

function [new_picture,new_hist]=hist_equal(picture,hist)%直方图均衡化
    Y=hist(1,:)/(size(picture,1)*size(picture,2));%将得到的直方图求每个灰度值所对应的概率
    for i=2:length(hist(1,:))
        Y(i)=Y(i-1)+Y(i);      %这部分是变成概率密度函数，就是进行概率的累加
    end   
    Y=Y*255;%乘以灰度值255
    Y=round(Y);%取整
    new_picture=ones(size(picture,1),size(picture,2));%建立新的图像
    for i=1:size(picture,1)
        for j=1:size(picture,2)
            new_picture(i,j)=Y(picture(i,j)+1);   %同样，由于灰度图像值是0~255，但索引为1~256，所以索引的时候需要编程picture(i，j)+1
        end
    end
    new_picture=uint8(new_picture);%由于上面的到的类型是double行，imshow的时候需要unit8型的变量
    new_hist=count_hist(new_picture);%均衡化后的直方图
end

function FFT_picture=FFT(picture)  %变换并且中心化
    G1=zeros(size(picture,1),size(picture,1));   %前变换矩阵    报告已给出证明
    G2=zeros(size(picture,2),size(picture,2));   %后变换矩阵
    picture1=double(picture);   %将图像uint8格式转换成double型，便于进行复数相乘
    for i=1:size(picture,1)
        for j=1:size(picture,2)
            picture1(i,j)=picture1(i,j)*exp(complex(0,1)*2*pi*(i+j)/2);%对原图像进行中心化预处理
        end
    end
    for i=1:size(picture,1)   %前变换矩阵的赋值
        for j=1:size(picture,1)
            G1(i,j)=exp(-complex(0,1)*2*pi*(i-1)*(j-1)/size(picture,1));
        end
    end
    for i=1:size(picture,2)   %后变换矩阵的赋值
        for j=1:size(picture,2)
            G2(i,j)=exp(-complex(0,1)*pi*2*(i-1)*(j-1)/size(picture,2));
        end
    end
    FFT_picture=G1*picture1*G2;   %二维傅里叶变换并且进行了中心化，返回的是复数型矩阵，要是想进行显示...
                                  %需要在程序中运行   imshow(log(abs(FFT_picture)+1),[])
end

function IFFT_picture=IFFT(f_picture)  %逆变换并且退中心化
    G1=zeros(size(f_picture,1),size(f_picture,1));%前变换矩阵
    G2=zeros(size(f_picture,2),size(f_picture,2));%后变换矩阵
    for i=1:size(f_picture,1)   %前变换矩阵赋初值
        for j=1:size(f_picture,1)
            G1(i,j)=exp(complex(0,1)*2*pi*(i-1)*(j-1)/size(f_picture,1))/size(f_picture,1);
        end
    end
    for i=1:size(f_picture,2)  %后变换矩阵赋初值
        for j=1:size(f_picture,2)
            G2(i,j)=exp(complex(0,1)*pi*2*(i-1)*(j-1)/size(f_picture,2))/size(f_picture,2);
        end
    end
    IFFT_picture=G1*f_picture*G2;   %进行逆变换
    for i=1:size(f_picture,1)  %退中心化，但是这个时候的到的矩阵还是复数矩阵
        for j=1:size(f_picture,2)
            IFFT_picture(i,j)=IFFT_picture(i,j)/exp(complex(0,1)*2*pi*(i+j)/2);
        end
    end
%     IFFT_picture=uint8(IFFT_picture);  %变换成uint8格式，便于之间输出逆变换的图像
end

function HF_picture=HF(picture) %同态滤波的实现
    picture=double(picture);%对于原图像进行double化，方便后面进行复数运算
    LN_picture=log(picture+1);%取对数,防止灰度值为0导致无法计算
    LNF_picture=FFT(LN_picture);%进行傅里叶正变换
    m=size(picture,1);   %行列参数
    n=size(picture,2);
    H=zeros(m,n);
    D0=10;
    rh=1.95;
    rl=0.5;%参数是通过多次调试得出来的效果比较好的一种情况
    for i=1:m
        for j=1:n
            H(i,j)=(rh-rl)*(1-exp(-((i-m/2)^2+(j-n/2)^2))/D0^2)+rl;  %同态滤波器的实现
        end
    end
    HFLNF_picture=LNF_picture.*H;  %频域滤波
    IFFTHFLNF_picture=IFFT(HFLNF_picture);%傅里叶逆变化
    HF_picture=exp(IFFTHFLNF_picture)-1;%在时域取指数，由于最初取对数的时候+1，取指数的时候需要进行减一操作
    HF_picture=uint8(abs(HF_picture));%类型转换为八位无符号整数uint8
end

function J=Region_Growing(image)  %区域生长
    I=im2double(image);%image就是直接提取得到的RGB三通道图像，归一化
    I=rgb2gray(I);
    [M,N]=size(I);
    J=zeros(M,N);
    x=300;
    y=500;%原图像为450*800，人工从图像中选取了一个背景像素点
    seed=I(x,y);%最开始的灰度均值
    J(x,y)=1;%将该点标记为背景
    temp=1;%计数变量，当temp=0的时候，while循环终止，也就是说对于整个图像已经分割完成
    esplion=0.015;%灰度均值的误差
    while temp>0
        temp=0;
        for i=2:M-1%本例用的是3*3模板生长，于是必须使得不能越界
            for j=2:N-1
                if J(i,j)==1%对图像遍历的过程中，如果该点被标记了，那么就对他的邻域进行检测...
                    for m=-1:1  %如果邻域内的点可以划分为同一区域，却没有标记，执行标记操作
                        for n=-1:1
                            if J(i+m,j+n)==0&abs(I(i+m,j+n)-seed)<esplion
                                J(i+m,j+n)=1;
                                temp=temp+1;
                            end
                        end
                    end 
                end
            end
        end
        seed=sum(sum(I.*J))/sum(sum(J));%灰度均值是对于上一次操作结束后，整个区域的均值
                                        %如果每标记一个点就求均值，就会使得背景范围变大，图像失真严重
    end
end

