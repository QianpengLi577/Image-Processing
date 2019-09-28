clc,clear;
picture=imread('I:\A.jpg');%读取图像的信息
picture=rgb2gray(picture);%将图像信息进行灰度化
hist=count_hist(picture);%计算直方图
% stem(H(2,:),H(1,:),'.');%打印直方图
% ylim([0 2*10^4])%限制直方图的Y值的上界限
% imhist(Y)%matlab自带的直方图显示图
[new_picture,new_hist]=hist_equal(picture,hist);
% stem(new_hist(2,:),new_hist(1,:),'.');
% ylim([0 2*10^4])
FFT_picture=FFT(picture);
% imshow(log(abs(FFT_picture)+1),[])
IFFT_picture=IFFT(FFT_picture);

% F1=fft2(picture);
% F1=fftshift(F1);
HF_picture=HF(picture);
imshow(HF_picture)
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
    G1=zeros(size(picture,1),size(picture,1));   %前变换矩阵
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

function HF_picture=HF(picture)
    picture=double(picture);%对于原图像进行double化，方便后面进行复数运算
    LN_picture=log(picture+1);%取对数
    LNF_picture=FFT(LN_picture);%进行傅里叶正变换
    m=size(picture,1);
    n=size(picture,2);
    H=zeros(m,n);
    D0=10;
    rh=1.95;
    rl=0.5;
    c=1.5;
    for i=1:m
        for j=1:n
            H(i,j)=(rh-rl)*(1-exp(-c*((i-m/2)^2+(j-n/2)^2))/D0^2)+rl;  %高斯滤波器的实现
        end
    end
    HFLNF_picture=LNF_picture.*H;  %频域滤波
    IFFTHFLNF_picture=IFFT(HFLNF_picture);%傅里叶逆变化
    HF_picture=exp(IFFTHFLNF_picture)-1;%在时域取指数，由于最初取对数的时候+1，取指数的时候需要进行减一操作
    HF_picture=uint8(HF_picture);%类型转换为八位无符号整数uint8
end

