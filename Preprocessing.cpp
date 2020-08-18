#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include <cmath>
#include <iostream>

Preprocessing::Preprocessing() {}
Preprocessing::~Preprocessing() {}

void Preprocessing::regulateImage(cv::Mat& image, int num)
{
    image = image + abs(num);
}

void Preprocessing::removeBoundary(cv::Mat& image, int min)
{
    if (min < 0)
    {
        regulateImage(image,min);
    }
    
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<float>(i,j) < min) // < 0
            {
                image.at<float>(i,j) = min-1; //16.5745
            }
        }
    }
}

void Preprocessing::removeBoundary(cv::Mat& image, int t, int set)
{
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<float>(i,j) < t)
            {
                image.at<float>(i,j) = set;
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
    cv::dilate(image, max, cv::getStructuringElement(cv::MORPH_RECT, cv::Size (25, 25))); //30x30
    cv::Mat res;
    cv::compare(image, max, res,cv::CMP_GE);
    
    cv::imwrite("max.tif",res);
    return res;
}