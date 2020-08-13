#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include <iostream>

Preprocessing::Preprocessing() {}
Preprocessing::~Preprocessing() {}

void Preprocessing::removeBoundary(cv::Mat& image)
{
    double min;
    double max;
    cv::minMaxLoc(image, &min, &max);
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<float>(i,j) < 0)
            {
                image.at<float>(i,j) = 16.5745;
            }
        }
    }
}


cv::Mat Preprocessing::createMask(cv::Mat& image, uint8_t t, uint8_t set)
{
    cv::Mat mask = cv::Mat::ones(image.size(),image.type());
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<uint8_t>(i,j) <= t)
            {
                mask.at<uint8_t>(i,j) = 0;
            }
        }
    }
    return mask;
}

cv::Mat Preprocessing::applyFilter(cv::Mat& image)
{
    cv::Mat imageFiltered;
    cv::bilateralFilter(image,imageFiltered,9,75,75);
    return imageFiltered;
}

cv::Mat Preprocessing::applyThreshold(cv::Mat& image)
{
    cv::Mat imageThresh;
    cv::threshold(image,imageThresh,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
    return imageThresh;
}

cv::Mat Preprocessing::applyThreshold(cv::Mat& image, int t, int max)
{
    cv::Mat imageThresh;
    cv::threshold(image,imageThresh,t,max,cv::THRESH_BINARY);
    return imageThresh;
}

cv::Mat Preprocessing::applyDistanceTranform(cv::Mat& image)
{
    cv::Mat dist;
    cv::distanceTransform(image, dist, cv::DIST_C, 3);
    return dist;
}

cv::Mat Preprocessing::findLocalMax(cv::Mat& image)
{
    cv::Mat max;
    cv::dilate(image, max, cv::getStructuringElement(cv::MORPH_RECT, cv::Size (30, 30)));
    //cv::imwrite("max.tif",max);
    //cv::Mat imgResult;
    //imgResult.create(image.rows, image.cols, image.type());
    //image.copyTo(imgResult, cv::compare(image, max, cv::CMP_GE);
    cv::Mat res;
    cv::compare(image, max, res,cv::CMP_GE);
    
    cv::imwrite("max.tif",res);
    return res;
}

