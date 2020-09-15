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

cv::Mat Preprocessing::applyFilterGausian(cv::Mat& image)
{
    cv::Mat imageFiltered;
    cv::GaussianBlur(image,imageFiltered,cv::Size(5,5),2);
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
    cv::dilate(image, max, cv::getStructuringElement(cv::MORPH_RECT, cv::Size (28, 28))); //30x30 //25
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

void Preprocessing::regionGrowUtil(cv::Mat& image, int x, int y, uint8_t prevC, uint8_t newC) 
{ 
    // Base cases 
    if (x < 0 || x >= image.rows || y < 0 || y >= image.cols) 
        return; 
    if (image.at<uint8_t>(x,y) == newC)  
        return;
    uint8_t xx = image.at<uint8_t>(x,y);
    uint8_t prev = prevC;
    if ((prevC - image.at<uint8_t>(x,y)) > 30) //>15
        return; 
      
  
    // Replace the color at (x, y) 
    image.at<uint8_t>(x,y) = newC; 
  
    // Recur for north, east, south and west 
    regionGrowUtil(image, x+1, y, prevC, newC); 
    regionGrowUtil(image, x-1, y, prevC, newC); 
    regionGrowUtil(image, x, y+1, prevC, newC); 
    regionGrowUtil(image, x, y-1, prevC, newC); 
} 
  
// It mainly finds the previous color on (x, y) and 
// calls floodFillUtil() 
void Preprocessing::regionGrow(cv::Mat& image, int x, int y, uint8_t newC) 
{ 
    uint8_t prevC = image.at<uint8_t>(x,y); 
    regionGrowUtil(image, x, y, prevC, newC); 
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

cv::Mat Preprocessing::findEdges(cv::Mat& image)
{
    cv::Mat image1 = image.clone();
    image1.convertTo(image1,CV_16SC1);
    cv::Mat image2 = image1.clone();
    cv::Mat edges = cv::Mat::zeros(image.size(),CV_8UC1);

    cv::Canny(image1,image2,edges,45,50,true);
    return edges;
}

void Preprocessing::findNeighbours(cv::Mat& image, std::vector<int>* neighbours)
{
    for (size_t i = 1; i < image.rows-1; i++)
    {
        for (size_t j = 1; j < image.cols-1; j++)
        {
            if(image.at<int32_t>(i-1,j) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i-1,j));
            if(image.at<int32_t>(i+1,j) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i+1,j));
            if(image.at<int32_t>(i,j-1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i,j-1));
            if(image.at<int32_t>(i,j+1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i,j+1));
            if(image.at<int32_t>(i-1,j-1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i-1,j-1));
            if(image.at<int32_t>(i+1,j-1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i+1,j-1));
            if(image.at<int32_t>(i-1,j+1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i-1,j+1));
            if(image.at<int32_t>(i+1,j+1) != image.at<int32_t>(i,j))
                neighbours[image.at<int32_t>(i,j)].push_back(image.at<int32_t>(i+1,j+1));          
        }      
    }
}

void Preprocessing::sortCells(cv::Mat& image, std::vector<cv::Point>* cells)
{

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            cells[image.at<int32_t>(i,j)].push_back(cv::Point(i,j));
        }
    }
}

void Preprocessing::findCentroids(std::vector<cv::Point>* cells,cv::Point* centroids,int len)
{
    for(size_t i = 0; i < len; i++)
    {
        int avgX = 0;
        int avgY = 0;
        for (size_t j = 0; j < cells[i].size(); j++)
        {
            avgX += cells[i][j].x;
            avgY += cells[i][j].y;
        }
        centroids[i] = cv::Point(avgX/cells[i].size(),avgY/cells[i].size());      
    }
}