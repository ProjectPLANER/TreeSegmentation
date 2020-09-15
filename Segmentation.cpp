#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include "Watershed.h"
#include "Segmentation.h"
#include <set>
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
    p.removeBoundary(image,0); //16 for easy
    std::cout << "Boundary removed..." << std::endl;
    //image = image*2;

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

    //test
    /*cv::Mat gt = cv::imread("data/gt.tif",cv::IMREAD_UNCHANGED);
    for (size_t i = 0; i < gt.rows; i++)
    {
        for (size_t j = 0; j < gt.cols; j++)
        {
            //cv::Vec4b pix = gt.at<cv::Vec4b>(i,j);
            if(gt.at<uint8_t>(i,j) == 255)
            {
                image_8U.at<uint8_t>(i,j) = 255;
            }
        }
    }*/
    

    //endtest
    //cv::imwrite(file+"_8UC1.tif",image_8U);
    image_8U = p.applyFilterGausian(image_8U);
    cv::imwrite(file+"_8UC1.tif",image_8U);

    cv::Mat slicmask, result;
    int min_element_size = 25;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,cv::ximgproc::SLICO,30,10);
    slic->iterate();
        //if (min_element_size>0)
            //slic->enforceLabelConnectivity(min_element_size);

        // get the contours for displaying
    slic->getLabelContourMask(slicmask, false);
    cv::Mat ming = cv::Mat::zeros(image_8U.size(),CV_32S);
    std::cout << slic->getNumberOfSuperpixels() << std::endl;
    slic->getLabels(ming);

    //cv::imwrite("labels.tif",ming);

    cv::Mat slicOut = image_8U;
    cv::cvtColor(image_8U,slicOut,cv::COLOR_GRAY2RGB);
    for (size_t i = 0; i < slicmask.rows; i++)
    {
        for (size_t j = 0; j < slicmask.cols; j++)
        {
            if(slicmask.at<int8_t>(i,j) < 0)
            {
                slicOut.at<cv::Vec3b>(i,j)[0] = 0;
                slicOut.at<cv::Vec3b>(i,j)[1] = 0;
                slicOut.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    } 
    cv::bitwise_not(slicmask,slicmask);
    
    for (size_t i = 0; i < mask.rows; i++)
    {
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i,j) == 0)
            {
                slicmask.at<uint8_t>(i,j) = 0;
            }
        }
    }
    cv::erode(slicmask, slicmask, cv::Mat());
    
    std::vector<std::vector<cv::Point>> slicContours;
    cv::findContours(slicmask,slicContours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
    cv::Mat slicMarkers = cv::Mat::zeros(slicmask.size(), CV_32S);

    for (size_t i = 0; i < slicContours.size(); i++)
    {
        //unsigned char R = ((i+1) & 0xff);
        //unsigned char G = ((i+1) & 0xff00) >> 8;
        //unsigned char B = ((i+1) & 0xff0000) >> 16;
        drawContours(slicMarkers, slicContours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
        //drawContours(slicMarkers, slicContours, i, cv::Scalar(R,G,B,255), -1);
        //cv::drawContours(slicMarkers,slicContours,i,cv::Scalar(i+1));

        
    }
    //cv::imwrite("ming.tif",slicMarkers);
    //cv::circle(slicMarkers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    //std::vector<int> slicColours;
    //for (int i = 1; i < slicContours.size(); i++)
    //{
    //   slicColours.push_back(i);
    //}
    //for (int i = 0; i < slicMarkers.rows; i++)
    //{
    //    for (int j = 0; j < slicMarkers.cols; j++)
    //    {
    //        int index = slicmask.at<int>(i,j);
    //        if (index > 0 && index <= static_cast<int>(slicContours.size()))
    //        {
    //            slicMarkers.at<int>(i,j) = slicColours[index-1];
    //        }
    //    }
    //}
    cv::imwrite("slic.tif",slicMarkers);


    std::set<int32_t> cells;
    std::vector<uint8_t> mapping[slic->getNumberOfSuperpixels()];
    std::vector<cv::Point> coord[slic->getNumberOfSuperpixels()];

    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            mapping[ming.at<int32_t>(i,j)].push_back(image_8U.at<uint8_t>(i,j));
            coord[ming.at<int32_t>(i,j)].push_back(cv::Point(i,j));
            if(localMax.at<uint8_t>(i,j) == 255)
            {
                cells.insert(ming.at<int32_t>(i,j));
            }       
        }
    }

    for (int i = 0; i < slic->getNumberOfSuperpixels(); i++)
    {
        if (cells.find(i) == cells.end())
        {
            float var = p.variance(mapping[i]);
            if (var > 25)
            {
                cv::Point maxPos = p.getMaxPosition(mapping[i],coord[i]);
                //localMax.at<uint8_t>(maxPos.x,maxPos.y) = 255;
            }          
        }
    }

    

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

    

    
    

    cv::Mat testMask = cv::Mat::zeros(ming.size(),CV_8U);
    for (size_t i = 0; i < testMask.rows; i++)
    {
        for (size_t j = 0; j < testMask.cols; j++)
        {
            if(cells.find(ming.at<int32_t>(i,j)) != cells.end())
            {
                testMask.at<uint8_t>(i,j) = 255;
            }
        }
    }
    cv::imwrite("testSlicMask.tif",testMask);
    cv::Mat edges = p.findEdges(image_8U);
    cv::imwrite("edges.tif",edges);


    cv::Mat newImg = image_8UC3.clone();
    for (size_t i = 0; i < newImg.rows; i++)
    {
        for (size_t j = 0; j < newImg.cols; j++)
        {
            if (testMask.at<uint8_t>(i,j) == 0)
            {
                newImg.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            } 
        }
    }
    
    

    //Region grow
/*
    cv::Mat regionGrow;
    cv::cvtColor(newImg,regionGrow,cv::COLOR_RGB2GRAY);
    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            if(localMax.at<uint8_t>(i,j) == 255 && regionGrow.at<uint8_t>(i,j) != 0)
            {
                p.regionGrow(regionGrow,i,j,255);
            }       
        }
    }
    cv::imwrite("regionGrow.tif",regionGrow);
    cv::Mat regionGrowColour = cv::Mat::zeros(image_8U.size(),CV_8UC4);
    //cv::cvtColor(newImg,regionGrowColour,cv::COLOR_RGB2RGBA);
    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            if(regionGrow.at<uint8_t>(i,j) == 255)
            {
                regionGrowColour.at<cv::Vec4b>(i,j) = cv::Vec4b(0,0,255,100);
            }       
        }
    }
    cv::imwrite("regionGrowColour.tif",regionGrowColour);

*/
    //End region grow
    

    cv::imwrite("markers.tif",markers);
    cv::watershed(newImg,markers); //image_8UC3
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
    display(image_8UC3,markers,slicMarkers,localMax);
    cv::namedWindow("Final Result",cv::WINDOW_NORMAL);
    cv::resizeWindow("Final Result", 1920,1080);
    cv::imshow("Final Result", dst);
    //cv::waitKey(0);

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

void Segmentation::display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& slicBoundaries,cv::Mat& localMax)
{
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            if (slicBoundaries.at<int32_t>(i,j) < 1)
            {
                image.at<cv::Vec3b>(i,j)[0] = 0;
                image.at<cv::Vec3b>(i,j)[1] = 255;
                image.at<cv::Vec3b>(i,j)[2] = 0;
            }
            if (boundaries.at<int32_t>(i,j) <= 1)
            {
                image.at<cv::Vec3b>(i,j)[0] = 0;
                image.at<cv::Vec3b>(i,j)[1] = 0;
                image.at<cv::Vec3b>(i,j)[2] = 255;
            }
            //std::cout << localMax.at<int8_t>(i,j) << std::endl;
            if (localMax.at<uint8_t>(i,j) != 0)
            {
                image.at<cv::Vec3b>(i,j)[0] = 0;
                image.at<cv::Vec3b>(i,j)[1] = 255;
                image.at<cv::Vec3b>(i,j)[2] = 255;
            }
            
        } 
    }
    cv::imwrite("output/"+file+"_boundaries.tif",image);
}