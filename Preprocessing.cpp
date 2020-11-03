/**
 * @file Preprocessing.cpp
 * @brief The preprocessing definition file, which defines all methods requires to process DEMs before segmentaton can take place.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <cmath>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"

/**
 * @brief Construct a new Preprocessing:: Preprocessing object
 * 
 */
Preprocessing::Preprocessing() {}

/**
 * @brief Destroy the Preprocessing:: Preprocessing object
 * 
 */
Preprocessing::~Preprocessing() {}

/**
 * @brief Scales an image to eliminate negative pixel values.
 * 
 * @param image An input image
 * @param scaleFactor The number by which to scale
 */
void Preprocessing::regulateImage(cv::Mat& image, int scaleFactor)
{
    image = image + abs(scaleFactor);
}

/**
 * @brief Removes the null values that may occur in DEMs that are not rectangular.
 * 
 * @param image An input image
 * @param min The minimum value in the image
 */
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

/**
 * @brief \deprecated Removes the null values that may occur in DEMs that are not rectangular.
 * 
 * @param image An input image
 * @param limit The threshold of the image
 * @param newValue The new value to replace null values
 */
void Preprocessing::removeBoundary(cv::Mat& image, int limit, int newValue)
{
    //Experimental - not in use

    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<float>(i,j) < limit)
            {
                image.at<float>(i,j) = newValue;
            }
        }
    }
}

/**
 * @brief \deprecated Returns a binary mask of an image where pixels with a value equal to or above the limit are set to 1 otherwise, they are set to 0.
 * 
 * @param image An input image
 * @param limit The mask limit
 * @return cv::Mat 
 */
cv::Mat Preprocessing::createMask(cv::Mat& image, uint8_t limit)
{
    //Experimental - not in use

    cv::Mat mask = cv::Mat::ones(image.size(),image.type());
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ )
        {
            if(image.at<uint8_t>(i,j) <= limit)
            {
                mask.at<uint8_t>(i,j) = 0;
            }
        }
    }
    return mask;
}

/**
 * @brief Returns the result after applying a Gaussian filter to the image inputted.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::applyFilterGausian(cv::Mat& image)
{
    cv::Mat imageFiltered;
    cv::GaussianBlur(image,imageFiltered,cv::Size(5,5),2);
    return imageFiltered;
}

/**
 * @brief \deprecated Returns the result after applying a bilateral filter to the image inputted.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::applyFilterBilateral(cv::Mat& image)
{
    //Experimental - not in use

    cv::Mat imageFiltered;
    cv::bilateralFilter(image,imageFiltered,5,75,75);
    return imageFiltered;
}

/**
 * @brief \deprecated Returns the result after applying an Otsu threshold operation on an image to obtain a binary image where pixels values above the local limit (which is calculated) are set to maxValue and values below the local limit are set to 0.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::applyThreshold(cv::Mat& image, int maxValue)
{
    //Experimental - not in use

    cv::Mat imageThresh;
    cv::threshold(image,imageThresh,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
    return imageThresh;
}

/**
 * @brief Returns the result after applying a threshold operation on an image to obtain a binary image where pixels values above the limit are set to maxValue and values below the limit are set to 0.
 * 
 * @param image An input image
 * @param limit The threshold limit
 * @param maxValue The maximum value of the thresholded image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::applyThreshold(cv::Mat& image, int limit, int maxValue)
{
    cv::Mat imageThresh;
    cv::threshold(image, imageThresh, limit, maxValue, cv::THRESH_BINARY);
    return imageThresh;
}

/**
 * @brief \deprecated Returns the result after applying a distance transform operation to the image inputted.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::applyDistanceTranform(cv::Mat& image)
{
    //Experimental - not in use

    cv::Mat dist;
    cv::distanceTransform(image, dist, cv::DIST_C, 3);
    return dist;
}

/**
 * @brief Returns the result after identifying all significant local maxima in the image inputted.
 * 
 * @param image An input image
 * @param windowSize The 
 * @return cv::Mat 
 */
cv::Mat Preprocessing::findLocalMax(cv::Mat& image, int windowSize)
{
    cv::Mat max;
    cv::dilate(image, max, cv::getStructuringElement(cv::MORPH_RECT, cv::Size (25, 25))); //30x30 //25
    cv::Mat res;
    cv::compare(image, max, res,cv::CMP_GE);
    
    //cv::imwrite("max.tif",res);
    return res;
}

/**
 * @brief \deprecated Returns the result after identifying all local maxima in the image inputted.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::findLocalMax(cv::Mat& image)
{
    //Experimental - not in use
    
    bool** rows = new bool*[image.rows];
    rows[0] = new bool[image.cols];
    rows[0][0] = (image.at<float>(0,0) >= image.at<float>(0,1) && image.at<float>(0,0) >= image.at<float>(1,0)) ? true : false;
    for (int j = 1; j < image.cols-1; j++)
    {
        rows[0][j] = (image.at<float>(0,j) >= image.at<float>(0,j-1) && image.at<float>(0,j) >= image.at<float>(0,j+1) && image.at<float>(0,j) >= image.at<float>(1,j)) ? true : false;
    }
    rows[0][image.cols-1] = (image.at<float>(0,image.cols-1) >= image.at<float>(0,image.cols-2) && image.at<float>(0,image.cols-1) >= image.at<float>(1,image.cols-1)) ? true : false;
    
    for (int i = 1; i < image.rows-1; i++)
    {
        rows[i] = new bool[image.cols];
        rows[i][0] = (image.at<float>(i,0) >= image.at<float>(i,1) && image.at<float>(i,0) >= image.at<float>(i-1,0) && image.at<float>(i,0) >= image.at<float>(i+1,0)) ? true : false;
        for (int j = 1; j < image.cols-1; j++)
        {
            rows[i][j] = (image.at<float>(i,j) >= image.at<float>(i,j-1) && image.at<float>(i,j) >= image.at<float>(i,j+1) && image.at<float>(i,j) >= image.at<float>(i-1,j) && image.at<float>(i,j) >= image.at<float>(i+1,j)) ? true : false;
        }
        rows[i][image.cols-1] = (image.at<float>(i,image.cols-1) > image.at<float>(i,image.cols-2) && image.at<float>(i,image.cols-1) > image.at<float>(i-1,image.cols-1) && image.at<float>(i,image.cols-1) > image.at<float>(i+1,image.cols-1)) ? true : false;
    }
    
    rows[image.rows-1] = new bool[image.cols];
    rows[image.rows-1][0] = (image.at<float>(image.rows-1,0) >= image.at<float>(image.rows-1,1) && image.at<float>(image.rows-1,0) >= image.at<float>(image.rows-2,0)) ? true : false;
    for (int j = 1; j < image.cols-1; j++)
    {
        rows[image.rows-1][j] = (image.at<float>(image.rows-1,j) >= image.at<float>(image.rows-1,j-1) && image.at<float>(image.rows-1,j) >= image.at<float>(image.rows-1,j+1) && image.at<float>(image.rows-1,j) >= image.at<float>(image.rows-2,j)) ? true : false;
    }
    rows[image.rows-1][image.cols-1] = (image.at<float>(image.rows-1,image.cols-1) >= image.at<float>(image.rows-1,image.cols-2) && image.at<float>(image.rows-1,image.cols-1) >= image.at<float>(image.rows-2,image.cols-1)) ? true : false;
    
    cv::Mat localMax = cv::Mat::zeros(image.size(),CV_8U);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (rows[i][j])
            {
                localMax.at<uint8_t>(i,j) = 255;
            }        
        }
        delete[] rows[i];
    }
    delete[] rows;
    return localMax;
}

/**
 * @brief \deprecated Performs the flood-fill algorithm on an image, by recursing through pixels.
 * 
 * @param image An input image
 * @param x coordinate of the pixel
 * @param y coordinate of the pixel
 * @param prevValue the original value of the pixel
 * @param newValue the new value of the pixel
 */
void Preprocessing::floodFillUtil(cv::Mat& image, int x, int y, int prevValue, int newValue) 
{ 
    //Experimental - not in use

    //Base cases 
    if (x < 0 || x >= image.rows || y < 0 || y >= image.cols) 
        return; 
    if (image.at<int32_t>(x,y) != prevValue) 
        return; 
    if (image.at<int32_t>(x,y) == newValue)  
        return;  
  
    //Replace the color at (x, y) 
    image.at<int32_t>(x,y) = newValue; 
  
    //Recur for north, east, south and west 
    floodFillUtil(image, x+1, y, prevValue, newValue); 
    floodFillUtil(image, x-1, y, prevValue, newValue); 
    floodFillUtil(image, x, y+1, prevValue, newValue); 
    floodFillUtil(image, x, y-1, prevValue, newValue); 
} 
  
/**
 * @brief \deprecated Performs a flood-fill operation an on image. Calls floodFillUtil(cv::Mat& image, int x, int y, uint8_t prevValue, uint8_t newValue)
 * 
 * @param image An input image
 * @param x coordinate of the start of the flood-fill operation
 * @param y coordinate of the start of the flood-fill operation
 * @param newValue the new value of the pixel
 */
void Preprocessing::floodFill(cv::Mat& image, int x, int y, int newValue) 
{
    //Experimental - not in use

    int prevValue = image.at<int32_t>(x,y); 
    floodFillUtil(image, x, y, prevValue, newValue); 
}

/**
 * @brief \deprecated A flood-fill algortihm that performs the region growing operation on an image, by recursing through pixels.
 * 
 * @param image An input image
 * @param x coordinate of the pixel
 * @param y coordinate of the pixel
 * @param prevValue the original value of the pixel
 * @param newValue the new value of the pixel
 */
void Preprocessing::regionGrowUtil(cv::Mat& image, int x, int y, uint8_t prevValue, uint8_t newValue) 
{ 
    //Experimental - not in use

    //Base cases 
    if (x < 0 || x >= image.rows || y < 0 || y >= image.cols) 
        return; 
    if (image.at<uint8_t>(x,y) == newValue)  
        return;
    if ((prevValue - image.at<uint8_t>(x,y)) > 30)
        return;    
  
    //Replace the color at (x, y) 
    image.at<uint8_t>(x,y) = newValue; 
  
    //Recur for north, east, south and west 
    regionGrowUtil(image, x+1, y, prevValue, newValue); 
    regionGrowUtil(image, x-1, y, prevValue, newValue); 
    regionGrowUtil(image, x, y+1, prevValue, newValue); 
    regionGrowUtil(image, x, y-1, prevValue, newValue); 
} 
  
/**
 * @brief \deprecated Performs a region growing operation an on image. Calls regionGrowUtil(cv::Mat& image, int x, int y, uint8_t prevValue, uint8_t newValue)
 * 
 * @param image An input image
 * @param x coordinate of the start of the region growing operation
 * @param y coordinate of the start of the region growing operation
 * @param newValue the new value of affected pixels
 */
void Preprocessing::regionGrow(cv::Mat& image, int x, int y, uint8_t newValue) 
{ 
    //Experimental - not in use

    uint8_t prevValue = image.at<uint8_t>(x,y); 
    regionGrowUtil(image, x, y, prevValue, newValue); 
}

/**
 * @brief Returns the variance from a list of pixels.
 * 
 * @param pixelValues An input list of pixel values
 * @return float 
 */
float Preprocessing::getVariance(std::vector<uint8_t> pixelValues)
{
    float mean = 0;
    for (size_t i = 0; i < pixelValues.size(); i++)
    {
        mean += pixelValues[i];
    }
    mean /= pixelValues.size();
    
    float var = 0;
    for(int i = 0; i < pixelValues.size(); i++ )
    {
        var += (pixelValues[i] - mean) * (pixelValues[i] - mean);
    }
    var /= (pixelValues.size()-1);
    return var;
}

/**
 * @brief \deprecated Returns the highest value from a list of pixels.
 * 
 * @param pixelValues An input list of pixel values
 * @return uint8_t 
 */
uint8_t Preprocessing::getMax(std::vector<uint8_t> pixelValues)
{
    //Experimental - not in use

    uint8_t maxValue = 0;
    for (size_t i = 0; i < pixelValues.size(); i++)
    {
        if(maxValue < pixelValues[i])
            maxValue = pixelValues[i];
    }
    return maxValue;
}

/**
 * @brief Returns the coordinates of the pixel with the highest value.
 * 
 * @param pixelValues An input list of pixel values
 * @param pixelCoord An input list of corresponding pixel coordinates
 * @return cv::Point 
 */
cv::Point Preprocessing::getMaxPosition(std::vector<uint8_t> pixelValues, std::vector<cv::Point> pixelCoord)
{
    uint8_t maxValue = 0;
    cv::Point maxCoord;
    for (size_t i = 0; i < pixelValues.size(); i++)
    {
        if(maxValue < pixelValues[i])
        {
            maxValue = pixelValues[i];
            maxCoord = pixelCoord[i];
        }       
    }
    return maxCoord;
}

/**
 * @brief \deprecated Returns a binary image with edges detected (displayed as white pixels) from an image according to the Canny algortihm.
 * 
 * @param image An input image
 * @return cv::Mat 
 */
cv::Mat Preprocessing::findEdges(cv::Mat& image)
{
    //Experimental - not in use

    cv::Mat image1 = image.clone();
    image1.convertTo(image1,CV_16SC1);
    cv::Mat image2 = image1.clone();
    cv::Mat edges = cv::Mat::zeros(image.size(),CV_8UC1);

    cv::Canny(image1,image2,edges,45,50,true);
    return edges;
}

/**
 * @brief Acquires the index of all neighbours of a cluster.
 * 
 * @param image An input image of type CV_32S containing clusters
 * @param neighbours An output vector containing the ID of every neighbour for every cluster
 */
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

/**
 * @brief Identifies all pixels in a cluster and arranges them in an array.
 * 
 * @param image An input image of clusters
 * @param clusters An output array of clusters
 */
void Preprocessing::sortCells(cv::Mat& image, std::vector<cv::Point>* clusters)
{
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            clusters[image.at<int32_t>(i,j)].push_back(cv::Point(i,j));
        }
    }
}

/**
 * @brief \deprecated Identifies the centroid pixel of every cluster in an image.
 * 
 * @param clusters An input image containing the clusters
 * @param centroids An output list of centroids
 * @param numCentroids The number of clusters in the image
 */
void Preprocessing::findCentroids(std::vector<cv::Point>* clusters, cv::Point* centroids, int numClusters)
{
    //Experimental - not in use

    for(size_t i = 0; i < numClusters; i++)
    {
        int avgX = 0;
        int avgY = 0;
        for (size_t j = 0; j < clusters[i].size(); j++)
        {
            avgX += clusters[i][j].x;
            avgY += clusters[i][j].y;
        }
        centroids[i] = cv::Point(avgX/clusters[i].size(),avgY/clusters[i].size());      
    }
}
