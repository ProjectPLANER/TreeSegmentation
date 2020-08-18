#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include "Watershed.h"
#include "Segmentation.h"
#include <iostream>


Segmentation::Segmentation(cv::Mat& image, std::string file) 
{
    this->image = image;
    this->file = file;
}

Segmentation::~Segmentation() {}

void Segmentation::segment()
{
    Preprocessing p;
    cv::Mat mask = p.applyThreshold(image,0,1); //easy: (image,0,1); medium:(image,-2,1); hard:(image,19,1)
    mask.convertTo(mask,CV_8U);
    //p.removeBoundary(image,19,0);
    std::cout << "Mask image created..." << std::endl;
    p.removeBoundary(image,16);
    std::cout << "Boundary removed..." << std::endl;
    //hard
    /*for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            //image.at<float>(i,j) = image.at<float>(i,j) * image.at<float>(i,j) * image.at<float>(i,j);
        }
    }*/

    cv::Mat localMax = p.findLocalMax(image);
    //cv::bitwise_and(mask, localMax, localMax);
    //cv::imwrite("bit.tif",localMax);
    for (size_t i = 0; i < mask.rows; i++)
    {
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i,j) == 0)
            {
                localMax.at<uint8_t>(i,j) = 0;
            }
        }
    }
    std::cout << "Local maximum calculated..." << std::endl;

    cv::Mat image_8U;
    cv::normalize(image, image_8U, 0, 255, cv::NORM_MINMAX);
    image_8U.convertTo(image_8U,CV_8U);
    //cv::Mat thresh = p.applyThreshold(image_8U);
    //cv::imwrite("thresh.tif",thresh);

    cv::Mat image_8UC3;
    cv::cvtColor(image_8U,image_8UC3,cv::COLOR_GRAY2RGB);
    std::cout << "DEM cleaned-up..." << std::endl;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(localMax, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
     // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(localMax.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    std::cout << "Contours calculated..." << std::endl;

    cv::watershed(image_8UC3,markers);
    //cv::Mat image_32SC3;
    //cv::normalize(image, image_32SC3, 0, 2147483647, cv::NORM_MINMAX);
    //cv::cvtColor(image_32SC3, image_32SC3, cv::COLOR_GRAY2RGB);
    //image_32SC3.convertTo(image_32SC3, CV_32SC3);
    //cv::imwrite("img.tif",image_32SC3);
    //Watershed w;
    //w.water(image_32SC3,markers);
    std::cout << "Watershed complete..." << std::endl;

    cv::imwrite("output/"+file+"_watershedLines.tif",markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark);
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    cv::imwrite("output/"+file+"_watershedBasins.tif",dst);
    // Visualize the final image
    display(image_8UC3,markers,localMax);
    cv::namedWindow("Final Result",cv::WINDOW_NORMAL);
    cv::resizeWindow("Final Result", 1920,1080);
    cv::imshow("Final Result", dst);
    cv::waitKey(0);

    /*cv::Mat otsu = p.applyThreshold(image_8U);
    cv::Mat dist = p.applyDistanceTranform(otsu);
    dist.convertTo(dist,CV_8U);
    cv::Mat dist_otsu = p.applyThreshold(dist,12,255);
    //cv::imwrite("dist.tif",dist_otsu);
    dist_otsu.convertTo(dist_otsu,CV_32S);
    cv::normalize(dist_otsu, dist_otsu, 0, 1, cv::NORM_MINMAX);
    cv::imwrite("dist2.tif",dist_otsu);
    cv::watershed(image_8UC3,dist_otsu);
    cv::imwrite("dist.tif",dist_otsu);*/
}

void Segmentation::display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& localMax)
{
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            if (boundaries.at<int32_t>(i,j) <= 1)
            {
                image.at<cv::Vec3b>(i,j)[0] = 0;
                image.at<cv::Vec3b>(i,j)[1] = 0;
                image.at<cv::Vec3b>(i,j)[2] = 255;
            }
            //std::cout << localMax.at<int8_t>(i,j) << std::endl;
            if (localMax.at<int8_t>(i,j) != 0)
            {
                image.at<cv::Vec3b>(i,j)[0] = 0;
                image.at<cv::Vec3b>(i,j)[1] = 255;
                image.at<cv::Vec3b>(i,j)[2] = 255;
            }
        } 
    }
    cv::imwrite("output/"+file+"_boundaries.tif",image);
}