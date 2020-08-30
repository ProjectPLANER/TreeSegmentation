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

// A recursive function to replace previous color 'prevC' at  '(x, y)'  
// and all surrounding pixels of (x, y) with new color 'newC' and 
void Preprocessing::floodFillUtil(cv::Mat& image, int x, int y, int prevC, int newC) 
{ 
    // Base cases 
    if (x < 0 || x >= image.rows || y < 0 || y >= image.cols) 
        return; 
    if (image.at<int32_t>(x,y) != prevC) 
        return; 
    if (image.at<int32_t>(x,y) == newC)  
        return;  
  
    // Replace the color at (x, y) 
    image.at<int32_t>(x,y) = newC; 
  
    // Recur for north, east, south and west 
    floodFillUtil(image, x+1, y, prevC, newC); 
    floodFillUtil(image, x-1, y, prevC, newC); 
    floodFillUtil(image, x, y+1, prevC, newC); 
    floodFillUtil(image, x, y-1, prevC, newC); 
} 
  
// It mainly finds the previous color on (x, y) and 
// calls floodFillUtil() 
void Preprocessing::floodFill(cv::Mat& image, int x, int y, int newC) 
{ 
    int prevC = image.at<int32_t>(x,y); 
    floodFillUtil(image, x, y, prevC, newC); 
}

float Preprocessing::variance(std::vector<uint8_t> cell)
{
    float mean = 0;
    for (size_t i = 0; i < cell.size(); i++)
    {
        mean += cell[i];
    }
    mean /= cell.size();
    
    float var = 0;
    for(int i = 0; i < cell.size(); i++ )
    {
        var += (cell[i] - mean) * (cell[i] - mean);
    }
    var /= (cell.size()-1);
    return var;
}

uint8_t Preprocessing::getMax(std::vector<uint8_t> cell)
{
    uint8_t max = 0;
    for (size_t i = 0; i < cell.size(); i++)
    {
        if(max < cell[i])
            max = cell[i];
    }
    return max;
}

cv::Point Preprocessing::getMaxPosition(std::vector<uint8_t> val,std::vector<cv::Point> coord)
{
    uint8_t max = 0;
    cv::Point maxPos;
    for (size_t i = 0; i < val.size(); i++)
    {
        if(max < val[i])
        {
            max = val[i];
            maxPos = coord[i];
        }
            
    }
    return maxPos;
}