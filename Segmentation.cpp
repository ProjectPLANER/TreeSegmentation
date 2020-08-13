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
    cv::cvtColor(easyMark,easyMark,cv::COLOR_RGBA2GRAY);
    easyMark.convertTo(easyMark_32S,CV_32S);
    for (size_t i = 0; i < easyMark_32S.rows; i++)
    {
        for (size_t j = 0; j < easyMark_32S.cols; j++)
        {
            if (easyMark_32S.at<int32_t>(i,j) < 255)
            {
                easyMark_32S.at<int32_t>(i,j) = 0;
            }
            /*if (easyMark_32S.at<int32_t>(i,j) == 255)
            {
                easyMark_32S.at<int32_t>(i,j) = 2147483647;
            }*/
            
        }
        
    }
    easyMark_32S.convertTo(easyMark,CV_8U);
    //easyMark_32S.convertTo(easyMark_32S,CV_32S);
    cv::Mat norm = easyMark_32S.clone();
    //cv::normalize(easyMark_32S,norm,0,1,cv::NORM_MINMAX);
    //cv::imwrite("easy.png",easyMark_32S);


    /*std::vector<std::vector<cv::Point> > contours;
    cv::findContours(easyMark, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
     // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(easyMark.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    //imshow("Markers", markers*10000);
    //cv::imwrite("easy.png",markers);

    cv::Mat easy_8UC3;
    cv::cvtColor(easy,easy_8UC3,cv::COLOR_RGBA2RGB);
    cv::watershed(easy_8UC3,markers);
    cv::imwrite("easy.png",markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
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
    }*/
    // Visualize the final image
    //cv::imshow("Final Result", dst);
    //cv::waitKey(0);








    Preprocessing p;
    cv::Mat mask = p.applyThreshold(image,0,1);
    mask.convertTo(mask,CV_8U);
    p.removeBoundary(image);
    //cv::Mat imageFiltered = p.applyFilter(image);
    //cv::imwrite("filter.tif",mask);
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

    cv::Mat image_8U;
    cv::normalize(image, image_8U, 0, 255, cv::NORM_MINMAX);
    image_8U.convertTo(image_8U,CV_8U);
    cv::Mat image_8UC3;
    cv::cvtColor(image_8U,image_8UC3,cv::COLOR_GRAY2RGB);

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
    //imshow("Markers", markers*10000);
    //cv::imwrite("easy.png",markers);

    cv::watershed(image_8UC3,markers);
    cv::imwrite("easy.png",markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
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
    // Visualize the final image
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