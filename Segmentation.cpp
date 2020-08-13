#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include "Segmentation.h"
#include <iostream>

Segmentation::Segmentation(cv::Mat& image) 
{
    this->image = image;
}

Segmentation::~Segmentation() {}

void Segmentation::segment()
{
    cv::Mat easy = cv::imread("easy_crop.png",cv::IMREAD_UNCHANGED);
    cv::Mat easyMark = cv::imread("easy_crop_mark.png",cv::IMREAD_UNCHANGED);
    cv::Mat easyMark_32S;
    //std::cout << easyMark.type() << std::endl;
    cv::cvtColor(easyMark,easyMark_32S,cv::COLOR_RGBA2GRAY);
    for (size_t i = 0; i < easyMark_32S.rows; i++)
    {
        for (size_t j = 0; j < easyMark_32S.cols; j++)
        {
            if (easyMark_32S.at<uint8_t>(i,j) < 255)
            {
                easyMark_32S.at<uint8_t>(i,j) = 0;
            }
            
        }
        
    }
    easyMark_32S.convertTo(easyMark_32S,CV_32S);
    //cv::normalize(easyMark_32S,easyMark_32S,0,1,cv::NORM_MINMAX);
    //cv::imwrite("easy.png",easyMark_32S);

    cv::Mat easy_8UC3;
    cv::cvtColor(easy,easy_8UC3,cv::COLOR_RGBA2RGB);
    cv::watershed(easy_8UC3,easyMark_32S);
    cv::imwrite("easy.png",easy_8UC3);








    Preprocessing p;
    p.removeBoundary(image);
    cv::Mat imageFiltered = p.applyFilter(image);
    //cv::imwrite("filter.tif",imageFiltered);
    p.findLocalMax(image);
    cv::Mat image_8U;
    cv::normalize(image, image_8U, 0, 255, cv::NORM_MINMAX);
    image_8U.convertTo(image_8U,CV_8U);
    cv::Mat mask = p.createMask(image_8U,10,0);
    cv::Mat max = p.findLocalMax(image_8U);
    for (size_t i = 0; i < image_8U.rows; i++)
    {
        for (size_t j = 0; j < image_8U.cols; j++)
        {
            max.at<uint8_t>(i,j) = max.at<uint8_t>(i,j) * mask.at<uint8_t>(i,j);
        }
        
    }
    cv::imwrite("fff.tif",max);
    

    cv::Mat image_8UC3;
    cv::cvtColor(image_8U,image_8UC3,cv::COLOR_GRAY2RGB);

    cv::Mat otsu = p.applyThreshold(image_8U);
    cv::Mat dist = p.applyDistanceTranform(otsu);
    dist.convertTo(dist,CV_8U);
    cv::Mat dist_otsu = p.applyThreshold(dist,12,255);
    //cv::imwrite("dist.tif",dist_otsu);
    dist_otsu.convertTo(dist_otsu,CV_32S);
    cv::normalize(dist_otsu, dist_otsu, 0, 1, cv::NORM_MINMAX);
    cv::imwrite("dist2.tif",dist_otsu);
    cv::watershed(image_8UC3,dist_otsu);
    cv::imwrite("dist.tif",dist_otsu);


}